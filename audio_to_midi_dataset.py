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
MIDI_EVENT_VOCCAB_SIZE = 90

MAX_EVENT_TIMESTAMP = 5.0
SEQUENCE_START = 1
SEQUENCE_END = 0
ACTIVE_EVENT_SEPARATOR = 2
BLANK_MIDI_EVENT = -1
BLANK_VELOCITY = 0
BLANK_DURATION = 0
NUM_VELOCITY_CATEGORIES = 10
FRAME_BLANK_VALUE = 0

SAMPLES_PER_FFT = 2 ** 11
WINDOW_OVERLAP = 0.90
COMPRESSION_FACTOR = 1
FREQUENCY_CUTOFF = 4000
LINEAR_SCALING = 10

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


# TODO: Before using this again, we need to check if the time-shifting causes the attack_time + duration
#       to be outside the bound of the audio.
@partial(jax.jit, static_argnames=["duration_per_frame"], donate_argnames=["frames"])
def time_shift_audio_and_events(
    duration_per_frame: float,
    frames: Float[Array, "num_samples"],
    midi_events: Integer[Array, "num_events 3"],
    key: jax.random.PRNGKey,
):
    """
    In order to introduce more diversity in the training data, we will offset the audio signal and media event
    by a randomly picked amount
    """
    # If as a result of the time-shift a midi event happens just before time 0 (defined by `epsilon` seconds before 0.0),
    # we will still include the event and reset its position to 0. This is because there's usually a slight delay in the
    # audio samples, and the FFT will aggregate information over a short time period.
    # This mainly occours when predicting the start of a new audio file and the audio starts right away (frequent in the training data).
    epsilon = 0  # frames

    # Due to active events, for now we only shift audio to the right
    offset_amounts_in_frames = jnp.round(jax.random.uniform(key, shape=(1,), minval=0, maxval=10)).astype(jnp.int16)

    # Handle audio samples
    audio_frame_positions = jnp.arange(frames.shape[0])
    frame_mask = audio_frame_positions < jnp.absolute(offset_amounts_in_frames)
    frame_mask = jnp.roll(
        frame_mask, shift=jnp.min(jnp.array([offset_amounts_in_frames, jnp.zeros(1)]))
    )  # Flip the mask if `offset_amounts_in_frames`` is negative
    frame_mask = jnp.repeat(frame_mask[:, None], repeats=frames.shape[1], axis=1)
    frames = jnp.roll(frames, shift=offset_amounts_in_frames, axis=0)
    frames = jnp.select(
        [~frame_mask],
        [frames],
        FRAME_BLANK_VALUE,
    )

    # Handle midi events
    # jax.debug.print("Original events = {midi_events}", midi_events=midi_events)

    positions = jnp.arange(midi_events.shape[0])
    updated_midi_event_times = midi_events[:, 0] + offset_amounts_in_frames
    # Special case: Update the timestamps happening `epsilon` seconds before 0.0 to make them appear at
    #               frame 0, per the reasoning in the start of the function.
    updated_midi_event_times = jnp.select(
        [(updated_midi_event_times < 0) & (updated_midi_event_times > -epsilon)],
        [0],
        updated_midi_event_times,
    )

    end_of_sequence = jnp.where(midi_events[:, 1] == SEQUENCE_END, size=1)[0]
    positions_to_update = (
        (positions > 0)
        & (positions < end_of_sequence)
        & (updated_midi_event_times >= 0)
        & (updated_midi_event_times < MAX_EVENT_TIMESTAMP / duration_per_frame)
    )
    first_to_keep = jnp.where(positions_to_update == True, size=1)[0]
    num_to_keep = jnp.sum(positions_to_update)

    # Skip all the events that will be prior to time 0 (minus the first because it is start of sequence)
    updated_midi_event_times = jnp.roll(
        updated_midi_event_times, -first_to_keep + 1, axis=0
    )
    positions_to_update = jnp.roll(positions_to_update, -first_to_keep + 1, axis=0)
    rolled_midi_events = jnp.roll(midi_events, -first_to_keep + 1, axis=0)

    # jax.debug.print("Mask = {mask}", mask=positions_to_update)

    # Construct a blank input that will contain start of sequence, end of sequence and padding
    # Values from this array will be selected outside of the mask
    blank_input = jnp.tile(
        jnp.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=jnp.int16), (midi_events.shape[0], 1)
    )
    blank_input = blank_input.at[0, :].set(
        jnp.array([0, SEQUENCE_END, BLANK_VELOCITY], dtype=jnp.int16)
    )
    blank_input = jnp.roll(
        blank_input, num_to_keep + 1, axis=0
    )  # +1 due to start of sequence
    blank_input = blank_input.at[0, :].set(
        jnp.array([0, SEQUENCE_START, BLANK_VELOCITY], dtype=jnp.int16)
    )

    # jax.debug.print("Blank input = {blank_input}", blank_input=blank_input)

    # Apply the mask picking `updated_midi_event_timestamps` inside `positions_to_update` and from `blank_input` outside
    midi_events = midi_events.at[:, 0].set(
        jnp.select(
            [positions_to_update, positions >= 0],
            [updated_midi_event_times, blank_input[:, 0]],
        )
    )
    for i in range(1, 3):  # Loop over position, midi event, and velocity
        midi_events = midi_events.at[:, i].set(
            jnp.select(
                [positions_to_update, positions >= 0],
                [rolled_midi_events[:, i], blank_input[:, i]],
            )
        )

    # jax.debug.print("Updated events = {updated_events}", updated_events=midi_events)

    return frames, midi_events

def find_active_events(carry, event: Integer[Array, "3"]):
    index, active_events, released_keys = carry

    next_active_events = jnp.select([
        event[1] == BLANK_MIDI_EVENT,
        event[2] == BLANK_VELOCITY,
        ~released_keys[event[1]],
    ],
    [
        active_events, # Blank event
        active_events, # Release key event
        active_events.at[index].set(True), # Attack event, and it has not been released! It is active!
    ],
    active_events).astype(jnp.bool_)

    next_release_events = jnp.select([
        event[1] == BLANK_MIDI_EVENT,
        event[2] == BLANK_VELOCITY,
    ],
    [
        released_keys, # Blank event
        released_keys.at[event[1]].set(True), # Release key event
    ],
    released_keys).astype(jnp.bool_)

    return (index - 1, next_active_events, next_release_events), jnp.zeros(0)


@jax.jit
def get_active_events(seen_events: Float[Array, "max_len 3"]):
    # Find the indices of the active events within the `seen_events`array
    empty_state = (
        seen_events.shape[0] - 1, # index
        jnp.zeros(seen_events.shape[0], dtype=jnp.bool_), # (active event indicator
        jnp.zeros(MIDI_EVENT_VOCCAB_SIZE, dtype=jnp.bool_) # released keys indicator
    )
    (_, active_events, _), _ = jax.lax.scan(find_active_events, empty_state, seen_events, reverse=True)

    # Using the indices, select them, and group them together
    sorted_indices = jnp.argsort(active_events, kind='stable')
    num_active = jnp.count_nonzero(active_events)
    active_events = jnp.repeat(active_events[:, None], repeats=3, axis=1)
    blank_events = jnp.repeat(jnp.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=jnp.int16)[None, :], repeats=seen_events.shape[0], axis=0)
    actual_active_events = jnp.where(active_events, seen_events, blank_events).astype(jnp.int16)
    actual_active_events = actual_active_events[sorted_indices, ...]

    # Place the active events at the position they should be appended to the `seen_events` array
    # num_seen_events = jnp.count_nonzero(seen_events[:, 1] != BLANK_MIDI_EVENT)
    # +1 to allow for the split event
    # actual_active_events = jnp.roll(actual_active_events, shift=(num_active + num_seen_events + 1), axis=0)

    # Place the active events at the beginning
    actual_active_events = jnp.roll(actual_active_events, shift=num_active, axis=0)

    # jax.debug.print("Found active events: {active} for seen events {seen}", active=actual_active_events, seen=seen_events)

    # # Append the active events to the seen_events array
    # end_position = int(MAX_EVENT_TIMESTAMP / duration_per_frame_in_secs) + 1
    # actual_active_events = actual_active_events.at[:, 0].set(end_position) # Make all the non-closed events appear at the very end
    # picking_mask = jnp.repeat((actual_active_events[:, 1] != 0)[:, None], repeats=3, axis=1)
    # seen_events = jnp.where(picking_mask, actual_active_events, seen_events)
    # active_event_separator = jnp.array([end_position, ACTIVE_EVENT_SEPARATOR, 0], dtype=jnp.int16)
    # seen_events = seen_events.at[num_seen_events].set(active_event_separator)

    # jax.debug.print("Seen events after {seen}", seen=seen_events)

    return actual_active_events

@partial(jax.jit, static_argnames=["batch_size"])
@partial(jax.profiler.annotate_function, name="audio_to_midi_generate_batch")
def generate_batch(
    key: jax.random.PRNGKey,
    batch_size: int,
    duration_per_frame_in_secs: float,
    frame_width_in_secs: float,
    selected_midi_events: Integer[Array, "batch_size max_len 4"],
    selected_audio_frames: Float[Array, "batch_size frames"],
):
    (
        key,
        sample_key,
        midi_split_key,
        time_shift_key,
    ) = jax.random.split(key, num=4)
    # TODO: Re-structure these transformations in a nicer way
    # timeshift_keys = jax.random.split(time_shift_key, num=selected_audio_frames.shape[0])
    # selected_audio_frames, selected_midi_events = jax.vmap(
    #     time_shift_audio_and_events, (None, 0, 0, 0)
    # )(duration_per_frame_in_secs, selected_audio_frames, selected_midi_events, timeshift_keys)

    sample_keys = jax.random.split(sample_key, num=batch_size)
    selected_audio_frames = jax.vmap(perturb_audio_frames, (0, 0))(
        selected_audio_frames, sample_keys
    )

    midi_event_counts = jnp.count_nonzero(
        selected_midi_events != BLANK_MIDI_EVENT, axis=(1)
    )[
        :, 1
    ]  # Count along the event id dimension and exclude any padded events with value BLANK_MIDI_EVENT, and pick any split pointing to the index to split _before_
    picked_midi_splits = jax.random.randint(
        midi_split_key,
        midi_event_counts.shape,
        minval=jnp.ones(
            midi_event_counts.shape
        ),  # We use ones for min value to make sure we do not pick the start of sequence
        maxval=midi_event_counts,
    )

    next_events = selected_midi_events[
        jnp.arange(selected_midi_events.shape[0]), picked_midi_splits
    ]

    event_indices = jnp.arange(selected_midi_events.shape[1])
    seen_event_mask = event_indices < picked_midi_splits[:, jnp.newaxis]
    seen_event_mask = jnp.repeat(seen_event_mask[:, :, None], repeats=4, axis=2) # Repeat for every (position, event, velocity) pair
    blank_events = jnp.repeat(jnp.repeat(jnp.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], dtype=jnp.int16)[None, :], selected_midi_events.shape[1], axis=0)[None, ...], batch_size, axis=0)
    seen_events = jnp.select(
        [~seen_event_mask],
        [blank_events],
        selected_midi_events
    )

    # Compute the set of active events so it can be used by the model
    # active_events = jax.vmap(get_active_events)(seen_events)

    # We can get rid of events that are BLANK_MIDI_EVENT for all samples in the batch
    # TODO: For now do not do this, as it leads to JAX recompilation pauses during the initial training steps
    # seen_events = seen_events[:, 0 : jnp.max(picked_midi_splits)]

    return {
        "audio_frames": selected_audio_frames,
        "seen_events": seen_events,
        "next_event": next_events,
        "duration_per_frame_in_secs": duration_per_frame_in_secs,
        "frame_width_in_secs": frame_width_in_secs,
        # "active_events": active_events,
    }


class AudioToMidiDatasetLoader:
    SAMPLE_RATE = 2 * FREQUENCY_CUTOFF # Sample rate to allow frequencies up to cutoff

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

        self.loaded_sample_names, self.loaded_midi_events, self.loaded_audio_frames, self.duration_per_frame, self.frame_width = self._load_random_samples(
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

            # Computing these selected samples in Python is super slow, and results in bottle necks
            # during training. If this is really important, do it in Rust...
            # selected_sample_names = [ current_loaded_sample_names[i] for i in indexes ]

            batch = generate_batch(
                batch_key,
                self.batch_size,
                self.duration_per_frame,
                self.frame_width,
                selected_midi_events,
                selected_samples,
            )

            # It is a bit hacky to inject this, but we can not send strings to JAX jit'ted functions
            # batch["sample_names"] = selected_sample_names

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
        audio_samples, midi_events = rust_plugins.load_events_and_audio(str(dataset_dir), samples, AudioToMidiDatasetLoader.SAMPLE_RATE, MAX_EVENT_TIMESTAMP, duration_per_frame)
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

        midi_events = AudioToMidiDatasetLoader._pad_and_stack_midi_events(midi_events, minimum_midi_event_size)

        if abs(calculated_duration_per_frame - duration_per_frame) > 0.001:
            raise CalculatedFrameDurationInvalid(calculated_duration_per_frame, duration_per_frame)
        return midi_events, frames, calculated_duration_per_frame, frame_width

    def _load_random_samples(self, num_samples_to_load: int, minimum_midi_event_size: Optional[int] = None):
        picked_samples = random.sample(self.all_sample_names, min(num_samples_to_load, len(self.all_sample_names)))
        midi_events, frames, calculated_duration_per_frame, frame_width = self.load_samples(self.dataset_dir, picked_samples, minimum_midi_event_size, self.sharding)
        return picked_samples, midi_events, frames, calculated_duration_per_frame, frame_width

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
                    current_frames = self.loaded_audio_frames
                    current_sample_names = self.loaded_sample_names

                # print(f"Loading {num_samples_to_load}, current size is {current_events.shape[0]}")
                sample_names, loaded_midi_events, loaded_audio_frames, _duration_per_frame, _frame_width = self._load_random_samples(num_samples_to_load, current_events.shape[1])

                current_amount_to_evict = max(0, num_samples_to_load - (num_samples_to_maintain - current_events.shape[0]))
                current_events = current_events[current_amount_to_evict:, ...]
                current_frames = current_frames[current_amount_to_evict:, ...]

                # I am intentionally using `np` here and not `jnp` so the maintained frames are only on the host
                blank_event = np.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], dtype=np.int16)
                current_events = np.concatenate([ # Make sure there is sufficient edge padding on the current events
                    current_events,
                    np.repeat(
                        np.repeat(blank_event[None, :], repeats=loaded_midi_events.shape[1] - current_events.shape[1], axis=0)[None, ...],
                        repeats=current_events.shape[0], axis=0)
                ], axis=1)

                new_events = np.concatenate([
                    current_events,
                    loaded_midi_events,
                ], axis=0)
                new_frames = np.concatenate([
                    current_frames,
                    loaded_audio_frames,
                ], axis=0)
                new_sample_names = sample_names + current_sample_names[current_amount_to_evict:]

                with self.sample_load_lock:
                    self.loaded_midi_events = new_events
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
    def _convert_samples(samples: Float[Array, "count samples"]):
        # Pad the signals with half the window size on each side to make sure the center of the Hann
        # window hits the full signal.
        padding_width = int(SAMPLES_PER_FFT / 2)
        padded_samples = jnp.pad(samples, ((0, 0), (padding_width, padding_width)), mode='constant', constant_values=0)
        frames = jax.vmap(fft_audio, (0, None, None))(padded_samples, SAMPLES_PER_FFT, WINDOW_OVERLAP)

        duration_per_frame = MAX_EVENT_TIMESTAMP / frames.shape[2]

        # Select only the lowest FREQUENCY_CUTOFF frequencies
        frame_width_in_secs = SAMPLES_PER_FFT / AudioToMidiDatasetLoader.SAMPLE_RATE
        cutoff_frame = int(FREQUENCY_CUTOFF * frame_width_in_secs)
        frames = frames[:, 0:cutoff_frame, :]

        return jnp.transpose(frames, axes=(0, 2, 1)), duration_per_frame, frame_width_in_secs

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
    sample_name: str, duration_per_frame: float, frame_width: float, frames: NDArray[jnp.float32]
):
    fig, ax1 = plt.subplots()
    X = jnp.linspace(0.0, duration_per_frame * frames.shape[0], frames.shape[0])
    Y = jnp.linspace(0.0, frames.shape[1] / frame_width, frames.shape[1])
    c = ax1.pcolor(X, Y, jnp.transpose(frames))

    ax1.set(
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
        title=f"Audio signal in frequency-domain\n{sample_name}",
    )
    fig.colorbar(c)

    ax2 = ax1.twiny()
    ax2.set_xlim(0, frames.shape[0])
    ax2.set_xlabel("Frame count")

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
    seen_events: Integer[Array, "seen_array_length 2"],
    next_event: Integer[Array, "4"],
    duration_per_frame_in_secs: float,
    frame_width: float,
):
    print(f"Sample name: {sample_name}")
    print("Frames shape:", frames.shape)
    print("Duration per frame:", duration_per_frame_in_secs)
    print("Frame width in seconds:", frame_width)
    print(f"Seen events: {_remove_zeros(seen_events)}")
    print(f"Next event: {next_event}")
    plot_frequency_domain_audio(sample_name, duration_per_frame_in_secs, frame_width, frames)
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
        dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/narrowed_keys_7"),
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
        print(f"Seen events shape {num}:", loaded_batch["seen_events"].shape)
        print(f"Next event shape: {num}", loaded_batch["next_event"].shape)

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
            "",
            loaded_batch["audio_frames"][batch_idx],
            loaded_batch["seen_events"][batch_idx],
            loaded_batch["next_event"][batch_idx],
            loaded_batch["duration_per_frame_in_secs"],
            loaded_batch["frame_width_in_secs"],
        )
