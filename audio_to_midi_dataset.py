import csv
import glob
from collections import deque
import random
import threading
import time
import numpy as np
from functools import partial
from pathlib import Path
import os

import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, Integer
from numpy.typing import NDArray
from typing import Optional, List
from threading import Lock
import math
import rust_plugins

# TODO: Clean this up
MIDI_EVENT_VOCCAB_SIZE = 88

MAX_EVENT_TIMESTAMP = 5.0
ACTIVE_EVENT_SEPARATOR = 2
BLANK_MIDI_EVENT = -1
BLANK_VELOCITY = 0
BLANK_DURATION = 0
NUM_VELOCITY_CATEGORIES = 10
FRAME_BLANK_VALUE = 0

SAMPLES_PER_FFT = 2 ** 13
WINDOW_OVERLAP = 0.80
COMPRESSION_FACTOR = 0.01
FREQUENCY_CUTOFF = 4000
LINEAR_SCALING = 150

def get_data_prep_config():
    return {
        "midi_voccab_size": MIDI_EVENT_VOCCAB_SIZE,
        "max_event_timestamp": MAX_EVENT_TIMESTAMP,
        "num_velocity_categories": NUM_VELOCITY_CATEGORIES,
        "samples_per_fft": SAMPLES_PER_FFT,
        "window_overlap": WINDOW_OVERLAP,
        "compression_factor": COMPRESSION_FACTOR,
        "frequency_cutoff": FREQUENCY_CUTOFF,
        "linear_scaling": LINEAR_SCALING
    }

@partial(jax.jit, donate_argnames=["frames"])
def perturb_audio_frames(
    frames, key: jax.random.PRNGKey
) -> Float[Array, "frames"]:
    """In order to make overfitting less likely this function perturbs the audio sampel in various ways:
    1. Add gausian noise
    """
    key1, key2 = jax.random.split(key, num=2)
    sigma = jax.random.uniform(key1) / 40  # Randomize the level of noise
    gaussian_noise = jnp.abs(sigma * jax.random.normal(key2, frames.shape))
    return frames + gaussian_noise


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

class CalculatedFrameDurationInvalid(Exception):
    def __init__(self, calculated, actual):
        super().__init__("Duration per frame mismatch")
        self.calculated_dpf = calculated
        self.actual_dpf = actual

# Overlap is in the percentage of the `window_size`` and should be in ]0;1[
@partial(jax.jit, static_argnames=["window_size", "overlap"])
def fft_audio(
    signal: NDArray[jnp.float32], window_size: int, overlap: float = 0.5
) -> NDArray[jnp.float32]:
    """Computes the spectrogram of an audio signal.
    """
    if window_size != next_power_of_2(window_size):
        raise "samples_per_fft must be a power of 2"
    hop_size = int(window_size * (1 - overlap))

    # Reshape the signal to match the expected input shape for conv_general_dilated_patches
    # The function expects (batch, spatial_dims..., features), so we add extra dimensions to fit
    signal = signal.reshape(1, -1, 1)  # Batch size = 1, 1 feature

    # Window the input signal and apply a Hann window
    hann_window = jnp.hanning(window_size)
    patches = jax.lax.conv_general_dilated_patches(
        lhs=signal,
        filter_shape=(window_size,),
        window_strides=(hop_size,),
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    windows = patches.squeeze(axis=(0,)) * hann_window

    # Apply the FFT
    fft = jax.vmap(jnp.fft.rfft)(windows)
    absolute_values = jnp.transpose(jnp.absolute(fft))

    # Do a logaritmic compression to emulate human hearing
    compressed_amplitudes = (
       jnp.sign(absolute_values)
       * jnp.log1p(COMPRESSION_FACTOR * jnp.abs(absolute_values))
       / jnp.log1p(COMPRESSION_FACTOR)
    )

    # Normalize the coefficients to give them closer to 0 mean based on some heuristic guided by the compression
    standardized_amplitudes = (compressed_amplitudes + FRAME_BLANK_VALUE) / LINEAR_SCALING

    return standardized_amplitudes

@partial(jax.jit, static_argnames=["batch_size"])
@partial(jax.profiler.annotate_function, name="audio_to_midi_generate_batch")
def generate_batch(
    key: jax.random.PRNGKey,
    batch_size: int,
    duration_per_frame_in_secs: float,
    frame_width_in_secs: float,
    selected_midi_events: Integer[Array, "batch_size frame_count midi_voccab_size"],
    selected_midi_events_human: Integer[Array, "batch_size num_events 4"],
    selected_audio_frames: Float[Array, "batch_size frames"],
    selected_sample_names = None,
):
    (
        key,
        sample_key,
    ) = jax.random.split(key, num=2)

    sample_keys = jax.random.split(sample_key, num=batch_size)
    selected_audio_frames = jax.vmap(perturb_audio_frames, (0, 0))(
        selected_audio_frames, sample_keys
    )

    return {
        "audio_frames": selected_audio_frames,
        "events": selected_midi_events,
        "events_human": selected_midi_events_human,
        "duration_per_frame_in_secs": duration_per_frame_in_secs,
        "frame_width_in_secs": frame_width_in_secs,
        "sample_names": selected_sample_names,
    }


class AudioToMidiDatasetLoader:
    SAMPLE_RATE = 44100

    def __init__(
        self,
        dataset_dir: Path,
        batch_size: int,
        prefetch_count: int,
        num_workers: int,
        key: jax.random.PRNGKey,
        num_samples_to_load: int = 250,
        num_samples_to_maintain: int = 5000,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count
        self.queue = deque([], prefetch_count + 1)
        self.sample_load_lock = Lock()
        self._stop_event = threading.Event()
        self._threads = []

        num_devices = len(jax.devices())
        device_mesh = mesh_utils.create_device_mesh((num_devices,))
        batch_mesh = Mesh(device_mesh, ("batch",))
        self.sharding = NamedSharding(
            batch_mesh,
            PartitionSpec(
                "batch",
            ),
        )

        self.all_sample_names = AudioToMidiDatasetLoader.load_sample_names(dataset_dir)

        self.loaded_sample_names, self.loaded_midi_events, self.loaded_midi_events_human, self.loaded_audio_frames, self.duration_per_frame, self.frame_width = self._load_random_samples(
            num_samples_to_load,
            minimum_midi_event_size=128)
        refresh_thread = threading.Thread(
            target=partial(self._periodic_refresh_samples, num_samples_to_load=num_samples_to_load, num_samples_to_maintain=num_samples_to_maintain),
            daemon=True,
        )
        self._threads.append(refresh_thread)
        refresh_thread.start()

        worker_keys = jax.random.split(key, num=num_workers)
        for worker_id in range(num_workers):
            worker_thread = threading.Thread(
                target=partial(self._generate_batch, key=worker_keys[worker_id]),
                daemon=True,
            )
            self._threads.append(worker_thread)
            worker_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Signal all threads to stop
        self._stop_event.set()
        for t in self._threads:
            t.join()

    def __iter__(self):
        while True:
            try:
                yield self.queue.popleft()
            except IndexError:
                print("WARNING: No elements in queue, should not really happen much...")
                time.sleep(0.1)

    def _generate_batch(self, key: jax.random.PRNGKey):
        while not self._stop_event.is_set():
            with self.sample_load_lock:
                current_loaded_midi_events = self.loaded_midi_events
                current_loaded_midi_events_human = self.loaded_midi_events_human
                current_loaded_audio_samples = self.loaded_audio_frames
                current_loaded_sample_names = self.loaded_sample_names

            batch_key, indexes_key, key = jax.random.split(key, num=3)
            indexes = jax.random.randint(
                indexes_key,
                shape=(self.batch_size,),
                minval=0,
                maxval=current_loaded_midi_events.shape[0],
                dtype=jnp.int32,
            )
            selected_samples = jax.device_put(current_loaded_audio_samples[indexes], self.sharding)
            selected_midi_events = jax.device_put(current_loaded_midi_events[indexes], self.sharding)
            selected_midi_events_human = jax.device_put(current_loaded_midi_events_human[indexes], self.sharding)

            # Computing these selected samples in Python is super slow, and results in bottle necks
            # during training. If this is really important, do it in Rust...
            selected_sample_names = [ current_loaded_sample_names[i] for i in indexes ]

            batch = generate_batch(
                batch_key,
                self.batch_size,
                self.duration_per_frame,
                self.frame_width,
                selected_midi_events,
                selected_midi_events_human,
                selected_samples,
            )

            # It is a bit hacky to inject this, but we can not send strings to JAX jit'ted functions
            batch["sample_names"] = selected_sample_names

            while len(self.queue) >= self.prefetch_count:
                # print("Backing off, as the queue is full")
                # Backoff mechanism
                time.sleep(0.05)
            self.queue.append(batch)

    def _pad_and_stack_midi_events(unpadded_midi_events: Integer[Array, "length 4"], minimum_midi_event_size: Optional[int] = None):
        max_events_length = max([events.shape[0] for events in unpadded_midi_events])
        if minimum_midi_event_size is not None:
            # We support extra padding to avoid JAX jit recompilations
            max_events_length = max(max_events_length, minimum_midi_event_size)
        padding = np.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], dtype=np.int16)[
            None, :
        ]
        padded_midi_events = [
            np.concatenate(
                [
                    events,
                    np.repeat(
                        padding, repeats=max_events_length - len(events), axis=0
                    ),
                ],
                axis=0,
            )
            for events in unpadded_midi_events
        ]
        stacked_events = np.stack(padded_midi_events, axis=0)
        return stacked_events

    @classmethod
    def load_samples(cls, dataset_dir: Path, samples: List[str], minimum_midi_event_size: Optional[int] = None, sharding = None):
        # We manually calculate the duration_per_frame vaiable here to be able to parallelize the event and sample loading
        # We ensure in the assert later that the computed value is indeed the correct one
        hop_size = (1 - WINDOW_OVERLAP) * SAMPLES_PER_FFT
        num_frames = math.ceil((AudioToMidiDatasetLoader.SAMPLE_RATE * MAX_EVENT_TIMESTAMP) / hop_size)
        duration_per_frame = MAX_EVENT_TIMESTAMP / num_frames
        audio_samples, midi_events_human, midi_events = rust_plugins.load_events_and_audio(str(dataset_dir), samples, AudioToMidiDatasetLoader.SAMPLE_RATE, MAX_EVENT_TIMESTAMP, duration_per_frame)
        audio_samples = jnp.stack(audio_samples)

        required_padding = 0
        if sharding is not None:
            # Make sure the audio_samples array is shardable
            required_padding = (sharding.mesh.shape["batch"] - audio_samples.shape[0]) % sharding.mesh.shape["batch"]
            if required_padding != 0:
                audio_samples = jnp.repeat(audio_samples, repeats=required_padding, axis=0)
            audio_samples = jax.device_put(audio_samples, sharding)

        frames, calculated_duration_per_frame, frame_width = AudioToMidiDatasetLoader._convert_samples(audio_samples)
        if required_padding > 0:
            frames = frames[:-required_padding, ...]

        midi_events_human = AudioToMidiDatasetLoader._pad_and_stack_midi_events(midi_events_human, minimum_midi_event_size)
        midi_events = jnp.stack(midi_events)

        if abs(calculated_duration_per_frame - duration_per_frame) > 0.001:
            raise CalculatedFrameDurationInvalid(calculated_duration_per_frame, duration_per_frame)
        return midi_events, midi_events_human, frames, calculated_duration_per_frame, frame_width

    def _load_random_samples(self, num_samples_to_load: int, minimum_midi_event_size: Optional[int] = None):
        picked_samples = random.sample(self.all_sample_names, min(num_samples_to_load, len(self.all_sample_names)))
        midi_events, midi_events_human, frames, calculated_duration_per_frame, frame_width = self.load_samples(self.dataset_dir, picked_samples, minimum_midi_event_size, self.sharding)
        return picked_samples, midi_events, midi_events_human, frames, calculated_duration_per_frame, frame_width

    def _periodic_refresh_samples(
        self,
        num_samples_to_load: int,
        num_samples_to_maintain: int,
        sleep_time: int = 0.1 # seconds
    ):
        # TODO: Consider doing this in a way that preserves determinism
        while True:
            self._stop_event.wait(sleep_time)
            if self._stop_event.is_set():
                return

            try:
                with self.sample_load_lock:
                    # Try to avoid unncessary JIT's due to different midi event lengths
                    current_events = self.loaded_midi_events
                    current_events_human = self.loaded_midi_events_human
                    current_frames = self.loaded_audio_frames
                    current_sample_names = self.loaded_sample_names

                # print(f"Loading {num_samples_to_load}, current size is {current_events.shape[0]}")
                sample_names, loaded_midi_events, loaded_midi_events_human, loaded_audio_frames, _duration_per_frame, _frame_width = self._load_random_samples(num_samples_to_load, current_events_human.shape[1])

                current_amount_to_evict = max(0, num_samples_to_load - (num_samples_to_maintain - current_events.shape[0]))
                current_events = current_events[current_amount_to_evict:, ...]
                current_frames = current_frames[current_amount_to_evict:, ...]

                # I am intentionally using `np` here and not `jnp` so the maintained frames are only on the host
                blank_event = np.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], dtype=np.int16)
                current_events_human = np.concatenate([ # Make sure there is sufficient edge padding on the current events
                    current_events_human,
                    np.repeat(
                        np.repeat(blank_event[None, :], repeats=loaded_midi_events_human.shape[1] - current_events_human.shape[1], axis=0)[None, ...],
                        repeats=current_events_human.shape[0], axis=0)
                ], axis=1)

                new_events = np.concatenate([
                    current_events,
                    loaded_midi_events,
                ], axis=0)
                new_events_human = np.concatenate([
                    current_events_human,
                    loaded_midi_events_human,
                ], axis=0)
                new_frames = np.concatenate([
                    current_frames,
                    loaded_audio_frames,
                ], axis=0)
                new_sample_names = sample_names + current_sample_names[current_amount_to_evict:]

                with self.sample_load_lock:
                    self.loaded_midi_events = new_events
                    self.loaded_midi_events_human = new_events_human
                    self.loaded_audio_frames = new_frames
                    self.loaded_sample_names = new_sample_names
            except CalculatedFrameDurationInvalid as e:
                calc_dpf = e.calculated_dpf
                actual_dpf = e.actual_dpf
                print(f"Calculated duration per frame {calc_dpf} does not line up with actual duration per frame {actual_dpf}." +
                    f"Difference was {abs(calc_dpf - actual_dpf)}. Trying again...")

    @classmethod
    def load_and_slice_full_audio(cls, filename: Path, overlap = 0.5):
        audio_samples = rust_plugins.load_full_audio(str(filename), AudioToMidiDatasetLoader.SAMPLE_RATE)

        window_size = round(MAX_EVENT_TIMESTAMP * AudioToMidiDatasetLoader.SAMPLE_RATE)
        overlap = round(overlap * AudioToMidiDatasetLoader.SAMPLE_RATE)

        step = window_size - overlap
        n_windows = math.ceil((audio_samples.shape[0] - overlap) / step)
        windows = []
        for i in range(n_windows):
            window_samples = audio_samples[i * step:i * step + window_size]
            # Make sure the window has the exact length (i.e. pad the last window if necessary)
            window_samples = jnp.pad(window_samples, ((0,window_size - window_samples.shape[0])), constant_values=(0,))
            windows.append(window_samples)
        windowed = jnp.stack(windows)
        
        return AudioToMidiDatasetLoader._convert_samples(windowed)


    @classmethod
    def load_audio_frames_from_sample_name(cls, dataset_dir: Path, sample_names: [str], sharding = None):
        # HACK: Consider getting rid of this approach by qualifying the sample names
        filenames = []
        audio_extensions = [ ".aac", ".aif" ]
        for sample in sample_names:
            found = False
            for extension in audio_extensions:
                candidate = dataset_dir / f"{sample}{extension}"
                if candidate.exists():
                    filenames.append(candidate)
                    found = True
                    break
            if not found:
                raise ValueError(f"Did not find audio file for sample named {sample}")

        return AudioToMidiDatasetLoader.load_audio_frames(filenames, sharding=sharding)

    @jax.jit
    def _convert_samples(samples: Float[Array, "count channel samples"]):
        # Pad the signals with half the window size on each side to make sure the center of the Hann
        # window hits the full signal.
        padding_width = int(SAMPLES_PER_FFT / 2)
        padded_samples = jnp.pad(samples, ((0, 0), (0,0), (padding_width, padding_width)), mode='constant', constant_values=0)
        left_frames = jax.vmap(fft_audio, (0, None, None))(padded_samples[:, 0, ...], SAMPLES_PER_FFT, WINDOW_OVERLAP)
        right_frames = jax.vmap(fft_audio, (0, None, None))(padded_samples[:, 1, ...], SAMPLES_PER_FFT, WINDOW_OVERLAP)

        duration_per_frame = MAX_EVENT_TIMESTAMP / left_frames.shape[2]

        # Select only the lowest FREQUENCY_CUTOFF frequencies
        frame_width_in_secs = SAMPLES_PER_FFT / AudioToMidiDatasetLoader.SAMPLE_RATE
        cutoff_frame = int(FREQUENCY_CUTOFF * frame_width_in_secs)
        left_frames = left_frames[:, 0:cutoff_frame, :]
        right_frames = right_frames[:, 0:cutoff_frame, :]

        # Make the frames on the form (batch, channel, temporal position, frequency)
        frames = jnp.transpose(jnp.stack([left_frames, right_frames]), axes=(1, 0, 3, 2))
        return frames, duration_per_frame, frame_width_in_secs

    @classmethod
    def load_sample_names(cls, dataset_dir: Path):
        audio_extensions = [ ".aif", ".aac" ]
        audio_names = set()
        for extension in audio_extensions:
            audio_names = audio_names.union(set(
                map(lambda path: path[(len(str(dataset_dir)) + 1):-4], glob.glob(f"{dataset_dir}/**/*{extension}", recursive=True))
            ))
        label_names = set(
            map(lambda path: path[(len(str(dataset_dir)) + 1):-4], glob.glob(f"{dataset_dir}/**/*.csv", recursive=True))
        )

        if audio_names != label_names:
            audio_no_csv = audio_names - label_names
            csv_no_audio = label_names - audio_names
            raise ValueError(f"Did not find the same set of labels and samples!, {audio_no_csv}, {csv_no_audio}")

        return list(sorted(audio_names))


def plot_time_domain_audio(sample_rate: int, samples: NDArray[jnp.float32]):
    time_indices = jnp.linspace(
        0, float(samples.size) / float(sample_rate), samples.size
    )

    fig, ax = plt.subplots()
    ax.plot(time_indices, samples)

    ax.set(
        xlabel="time (s)",
        ylabel="amplitude",
        title="Normalized audio signal in time-domain",
    )
    ax.grid()


def plot_frequency_domain_audio(
    sample_name: str, duration_per_frame: float, frame_width: float, frames: NDArray[jnp.float32], events: Integer[Array, "frame_count midi_voccab_size"] = None
):
    if events is None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

    left_frames = frames[0]
    X_left = jnp.linspace(0.0, duration_per_frame * left_frames.shape[0], left_frames.shape[0])
    Y_left = jnp.linspace(0.0, left_frames.shape[1] / frame_width, left_frames.shape[1])
    c_left = ax1.pcolor(X_left, Y_left, jnp.transpose(left_frames))

    right_frames = frames[0]
    X_right = jnp.linspace(0.0, duration_per_frame * right_frames.shape[0], right_frames.shape[0])
    Y_right = jnp.linspace(0.0, right_frames.shape[1] / frame_width, right_frames.shape[1])
    c_right = ax2.pcolor(X_right, Y_right, jnp.transpose(right_frames))

    ax1.set(
        ylabel="Frequency [Hz]",
        title=f"Audio signal in frequency-domain\n{sample_name}",
    )
    ax1.xaxis.set_visible(False)
    fig.colorbar(c_left)
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(0, frames.shape[1])
    ax1_twin.set_xlabel("Frame count")

    ax2.set_ylabel("Frequency [Hz]")
    if events is None:
        ax2.set_ylabel("Time [s]")
    fig.colorbar(c_right)

    if events is not None:
        X_events = jnp.linspace(0.0, duration_per_frame * left_frames.shape[0], left_frames.shape[0])
        Y_events = jnp.arange(MIDI_EVENT_VOCCAB_SIZE)
        c_events = ax3.pcolor(X_events, Y_events, jnp.transpose(events))
        ax3.set(
            xlabel="Time [s]",
            ylabel="MIDI Event",
        )
        fig.colorbar(c_events)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, top=0.90, bottom=0.08)

def plot_embedding(
    sample_name: str, embeddings: Float[Array, "frame_count embedding_size"]
):
    fig, ax1 = plt.subplots()
    X = jnp.arange(embeddings.shape[0]) + 1
    Y = jnp.arange(embeddings.shape[1])
    ax1.pcolor(X, Y, jnp.transpose(embeddings))

    ax1.set(
        xlabel="Frame",
        ylabel="Embedding",
        title=f"Audio frame embeddings\n{sample_name}",
    )

@jax.jit
def calculate_bin(frame, note_frequency, width_in_fft_bin, fft_bandwidth):
    fft_idx = note_frequency / fft_bandwidth
    # start_idx = jnp.int16(fft_idx - width_in_fft_bin)
    # length = jnp.int16(width_in_fft_bin * 2 + 1)
    start_idx = fft_idx
    length = 1
    mask = (jnp.arange(frame.shape[0]) >= start_idx) & (jnp.arange(frame.shape[0]) <= start_idx + length)
    return jnp.sum(frame * mask)

@jax.jit
def calculate_bins_for_frames(frames, note_frequencies, width_in_fft_bins, fft_bandwidth):
    @jax.jit
    def internal_calc(frame):
        return jax.vmap(calculate_bin, (None, 0, 0, None))(frame, note_frequencies[:-1], width_in_fft_bins, fft_bandwidth)
    return jax.vmap(internal_calc)(frames)

@jax.jit
def bin_audio_frames_to_notes(frames: Float[Array, "seq_len frame_size"]):
    bins = jnp.arange(99) + 1
    note_frequencies = jnp.power(2, (bins - 49) / 12) * 440
    # print(f"Note frequencies: {note_frequencies}")
    bin_widths = jnp.diff(note_frequencies) / 2
    # print(f"Bin widths: {bin_widths}")

    fft_bandwidth = FREQUENCY_CUTOFF / frames.shape[1]
    width_in_fft_bins = bin_widths / fft_bandwidth
    return calculate_bins_for_frames(frames, note_frequencies, width_in_fft_bins, fft_bandwidth)

def plot_with_frequency_normalization_domain_audio(
    sample_name: str, duration_per_frame: float, frame_width: float, frames: NDArray[jnp.float32]
):
    fig, ax1 = plt.subplots()

    binned = bin_audio_frames_to_notes(frames)
    X = jnp.linspace(0.0, duration_per_frame * frames.shape[0], frames.shape[0])
    Y = jnp.arange(binned.shape[1])
    ax1.pcolor(X, Y, jnp.transpose(binned))

    ax1.set(
        xlabel="Time [s]",
        ylabel="Note",
        title=f"Audio signal in frequency-domain\n{sample_name}",
    )

    ax2 = ax1.twiny()
    ax2.set_xlim(0, frames.shape[0])
    ax2.set_xlabel("Frame count")

def _remove_zeros(arr: Integer[Array, "len 4"]):
    blank_event = jnp.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], jnp.int16)
    mask = ~(jnp.all(arr == blank_event, axis=1))
    return arr[mask]

def visualize_sample(
    sample_name: str,
    frames: Float[Array, "num_samples"],
    events: Integer[Array, "num_frames midi_voccab_size"],
    events_human: Integer[Array, "num_events 4"],
    duration_per_frame_in_secs: float,
    frame_width: float,
):
    print(f"Sample name: {sample_name}")
    print("Frames shape:", frames.shape)
    print("Duration per frame:", duration_per_frame_in_secs)
    print("Frame width in seconds:", frame_width)
    print(f"Human evnets: {_remove_zeros(events_human)}")
    plot_frequency_domain_audio(sample_name, duration_per_frame_in_secs, frame_width, frames, events=events)
    # plot_with_frequency_normalization_domain_audio(sample_name, duration_per_frame_in_secs, frame_width, frames)

    plt.show()


def benchmark():
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v1")
    with open('dataset_benchmark.csv', mode='w', buffering=1) as benchmark_file:
        benchmark_csv = csv.writer(benchmark_file)

        for batch_size in [32,64,128,256,512,1024,2048,4096,8192]:
            for prefetch_count in [0,1,2,4,8]:
                for num_workers in [1,2,4,8]:
                    with AudioToMidiDatasetLoader(
                        dataset_dir=dataset_dir,
                        batch_size=batch_size,
                        prefetch_count=prefetch_count,
                        num_workers=num_workers,
                        key=key,
                    ) as dataset_loader:
                        dataset_loader_iter = iter(dataset_loader)

                        start_time = time.time()
                        generated_samples = 0
                        for num, loaded_batch in zip(range(0, 100), dataset_loader_iter):
                            generated_samples += loaded_batch["audio_frames"].shape[0]
                        finished_time = time.time()

                        benchmark_csv.writerow([batch_size, prefetch_count, num_workers, generated_samples, finished_time - start_time])


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update("jax_threefry_partitionable", True)
    key = jax.random.PRNGKey(42)

    # benchmark()

    # sample_rate, samples = load_audio_and_normalize(
    #     "/Volumes/git/ml/datasets/midi-to-sound/v0/piano_YamahaC7_68.aac"
    # )
    # sample_rate, samples = load_audio_and_normalize(
    #     "/Volumes/git/ml/datasets/midi-to-sound/v0/piano_YamahaC7_108.aac"
    # )
    # duration_per_frame, frequency_domain = fft_audio(sample_rate, samples)
    # duration_per_frame, frames = cleanup_fft_and_low_pass(
    #     duration_per_frame, frequency_domain
    # )

    # plot_time_domain_audio(sample_rate, samples)
    # plot_frequency_domain_audio(duration_per_frame, frames)

    # plt.show()

    # sample_rate, samples = load_audio_and_normalize(
    #     "/Volumes/git/ml/datasets/midi-to-sound/v0/piano_YamahaC7_68.aac"
    # )
    # perturbed_sampels = perturb_audio_sample(samples, key)

    # import sounddevice as sd

    # sd.play(perturbed_sampels, sample_rate)
    # sd.wait()

    # test that it sounds right (requires ffplay, or pyaudio):
    # from pydub.playback import play

    # play(audio_segment)

    # Test pretending we have multiple devices

    dataset_loader = AudioToMidiDatasetLoader(
        # dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set_only_yamaha"),
        # dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/debug"),
        # dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/debug_logic"),
        # dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/debug_logic_no_effects"),
        dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/dual_hands"),
        batch_size=1,
        prefetch_count=1,
        num_workers=1,
        key=key,
        num_samples_to_load=16,
        num_samples_to_maintain=200,
    )
    dataset_loader_iter = iter(dataset_loader)

    # agg_histogram = jnp.zeros(100, dtype=jnp.int32)
    for num, loaded_batch in zip(range(0, 200), dataset_loader_iter):
        print(f"Audio frames shape {num}:", loaded_batch["audio_frames"].shape)

        # seen_events = loaded_batch["seen_events"][:, :, 1] + 1 # +1 to make the blank event -1 appear as a 0
        # flattened_events = jnp.reshape(seen_events, (seen_events.shape[0] * seen_events.shape[1]))
        # special_mask = flattened_events < 3
        # print(f"Seen events max: {jnp.max(flattened_events)}")
        # print(f"Seen events min: {jnp.min(flattened_events[~special_mask])}")
        # histogram = jnp.bincount(flattened_events, length=agg_histogram.shape[0])
        # agg_histogram = agg_histogram + histogram
        # print(f"Histogram: {agg_histogram}")

        batch_idx = random.randint(0, loaded_batch["audio_frames"].shape[0] - 1)
        visualize_sample(
            loaded_batch["sample_names"][batch_idx],
            loaded_batch["audio_frames"][batch_idx],
            loaded_batch["events"][batch_idx],
            loaded_batch["events_human"][batch_idx],
            loaded_batch["duration_per_frame_in_secs"],
            loaded_batch["frame_width_in_secs"],
        )
