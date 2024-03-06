import csv
import glob
import queue
import random
import threading
import time
import numpy as np
from functools import partial
from pathlib import Path

import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, Integer
from numpy.typing import NDArray
from typing import Optional
from threading import Lock

import parallel_audio_reader

# TODO: Clean this up
MAX_EVENT_TIMESTAMP = 5.0
SEQUENCE_START = 1
SEQUENCE_END = 0
BLANK_MIDI_EVENT = -1
BLANK_VELOCITY = 0
NUM_VELOCITY_CATEGORIES = 10
FRAME_BLANK_VALUE = -(4/3)

@partial(jax.jit, donate_argnames=["frames"])
def perturb_audio_frames(
    frames, key: jax.random.PRNGKey
) -> Float[Array, "frames"]:
    """In order to make overfitting less likely this function perturbs the audio sampel in various ways:
    1. Add gausian noise
    """
    key1, key2 = jax.random.split(key, num=2)
    sigma = jax.random.uniform(key1) / 10  # Randomize the level of noise
    gaussian_noise = jnp.abs(sigma * jax.random.normal(key2, frames.shape))
    return frames + gaussian_noise


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@partial(jax.jit, static_argnames=["samples_per_fft"])
def fft_audio(
    samples: NDArray[jnp.float32], samples_per_fft: int
) -> NDArray[jnp.float32]:
    """Computes the fft of the audio samples
    Returns a tuple of the frame duration in seconds, and the complex FFT components

    Args:
        fft_duration: The duration in ms to compute the fft for. It will be rounded to the next
                      power of 2 samples
    """
    if samples_per_fft != next_power_of_2(samples_per_fft):
        raise "samples_per_fft must be a power of 2"

    num_padding_symbols = samples_per_fft - (samples.shape[0] % samples_per_fft)
    if num_padding_symbols == samples_per_fft:
        num_padding_symbols = 0
    padded_data = jnp.pad(samples, (0, num_padding_symbols), constant_values=0)
    data = padded_data.reshape(
        (int(padded_data.shape[0] / samples_per_fft), samples_per_fft)
    )
    fft = jnp.fft.fft(data)
    transposed_amplitudes = jnp.transpose(jnp.absolute(fft))

    # Do a logaritmic compression to emulate human hearing
    compression_factor = 255
    compressed_amplitudes = (
        jnp.sign(transposed_amplitudes)
        * jnp.log1p(compression_factor * jnp.abs(transposed_amplitudes))
        / jnp.log1p(compression_factor)
    )

    # Normalize the coefficients to give them closer to 0 mean based on some heuristic guided by the compression
    standardized_amplitudes = compressed_amplitudes + FRAME_BLANK_VALUE

    return standardized_amplitudes


def events_from_sample(
    dataset_dir: str, sample: str
) -> Float[Array, "max_number_of_events 3"]:
    """
    Returns a numpy array with tuples of (timestamp in seconds, midi event id, velocity)
    """
    epsilon = 0.05 # HACK: We want to make sure that because of numeic accuracy the release events
                   # comes before potential subsequent attach events
    def key_to_event(key: int):
        return 2 + (key - 21)

    events = []
    max_velocity = 0.0
    with open(f"{dataset_dir}/{sample}.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            attack_time = float(row[0])
            duration = float(row[1])
            key = int(row[2])
            velocity = float(row[3])

            # TODO: Make the event calculation nicer using the Midi dictionary
            # Notice we put releases prior to attacks in terms of midi event index
            if attack_time < MAX_EVENT_TIMESTAMP:
                events.append((attack_time, key_to_event(key) + 88, velocity))  # Attack
                max_velocity = max(max_velocity, velocity)
            if attack_time + duration < MAX_EVENT_TIMESTAMP + 5 * epsilon:
                # HACK: If `attack_time + duration > MAX_EVENT_TIMESTAMP` we will clamp it to `MAX_EVENT_TIMESTAMP`
                #       This has to do with the generated dataset in that it has a quirk that the release may be
                #       slightly beyond `MAX_EVENT_TIMESTAMP`, but in reality the note has been released
                #       I should be able to delete this once I fix and re-generate the dataset
                events.append(
                    (min(attack_time + duration - epsilon, MAX_EVENT_TIMESTAMP), key_to_event(key), BLANK_VELOCITY)
                )  # Release

    # Re-normalize velocities as we normalize the audio level as well
    events = [ (event[0], event[1], event[2] / max_velocity) for event in events ]

    # Append the SEQUENCE_START and SEQUENCE_EVENT events outside the sorting
    # TODO: Find a nicer way...
    return jnp.array(
        [(0.0, SEQUENCE_START, BLANK_VELOCITY)]
        + sorted(events)
        + [(0.0, SEQUENCE_END, BLANK_VELOCITY)]
    )


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
    epsilon = 2  # frames
    offset_amounts_in_frames = jnp.round(jax.random.uniform(key, shape=(1,), minval=-100, maxval=100)).astype(jnp.int16)

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


@partial(jax.jit, static_argnames=["duration_per_frame"])
def perturb_midi_event_positions(
    key: jax.random.PRNGKey,
    midi_events: Float[Array, "num_events 3"],
    duration_per_frame: float,
):
    """
    It is unlikely that we will have positions inferred completely correctly all the time, for two main reasons:
     1. Miliseconds of precision could be a bit skewed in training data and there may be a bit of variability between sampled instruments etc
     2. In the training objectibe function because of 1 we predict with a prob dist around the dataset position
    We will pick the new positions using a Gaussian prob distribution around the desired location, and make sure events
    maintain the same order as in the training dataset by taking the commulative max
    """
    pertubation = jnp.round(
        jax.random.normal(key, shape=(midi_events.shape[0],)) * 2 # Perturb with a normal distribution with a variance of 2 frames
    ).astype(jnp.int16)
    pertubation = pertubation + midi_events[:, 0]
    # Make sure the new perturbated event positions are monotonically increasing
    pertubation = jax.lax.cummax(pertubation)

    # Do not alter special events
    special_events = max(SEQUENCE_START, SEQUENCE_END, BLANK_MIDI_EVENT)
    adjusted_event_positions = jnp.select(
        [midi_events[:, 1] > special_events], [pertubation], midi_events[:, 0]
    )

    return midi_events.at[:, 0].set(adjusted_event_positions)


@partial(jax.jit, static_argnames=["duration_per_frame_in_secs"])
def event_positions_to_frame_time(
    selected_midi_events: Float[Array, "3"], duration_per_frame_in_secs: Float
) -> Integer[Array, "3"]:
    @jax.jit
    def convert_position_and_velocity(event: Float[Array, "3"]) -> Integer[Array, "3"]:
        """
        Positions and velocities are represented as floats in the dataset:
         - position: The number of seconds since the beginning as a float
         - velocity: A float between 0.0 and 1.0 indicating the velocity of the event
        We convert both of them to integers so they can be understood by the model.
        """
        position = jnp.round(event[0] / duration_per_frame_in_secs).astype(jnp.int16)
        velocity = jnp.round(event[2] * NUM_VELOCITY_CATEGORIES).astype(
            jnp.int16
        )  # Convert it into a group of `NUM_VELOCITY_CATEGORIES` velocities
        return jnp.array([position, event[1], velocity]).astype(jnp.int16)

    # Express midi event position in terms of frames and convert the array to an integer array
    return jnp.apply_along_axis(
        convert_position_and_velocity, 2, selected_midi_events
    ).astype(jnp.int16)


@partial(jax.jit, static_argnames=["batch_size", "duration_per_frame_in_secs"])
@partial(jax.profiler.annotate_function, name="audio_to_midi_generate_batch")
def generate_batch(
    key: jax.random.PRNGKey,
    batch_size: int,
    duration_per_frame_in_secs: float,
    selected_midi_events: Integer[Array, "batch_size max_len 3"],
    selected_audio_frames: Float[Array, "batch_size frames"],
):
    (
        key,
        sample_key,
        midi_split_key,
        time_shift_key,
        position_pertubation_key,
    ) = jax.random.split(key, num=5)
    # TODO: Re-structure these transformations in a nicer way
    timeshift_keys = jax.random.split(time_shift_key, num=selected_audio_frames.shape[0])
    selected_audio_frames, selected_midi_events = jax.vmap(
        time_shift_audio_and_events, (None, 0, 0, 0)
    )(duration_per_frame_in_secs, selected_audio_frames, selected_midi_events, timeshift_keys)

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
    seen_event_mask = jnp.repeat(seen_event_mask[:, :, None], repeats=3, axis=2) # Repeat for every (position, event, velocity) pair
    blank_events = jnp.repeat(jnp.repeat(jnp.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=jnp.int16)[None, :], selected_midi_events.shape[1], axis=0)[None, ...], batch_size, axis=0)
    seen_events = jnp.select(
        [~seen_event_mask],
        [blank_events],
        selected_midi_events
    )

    # We only perturb the seen events, and leave the next event to predict non-perturbed
    position_pertubation_keys = jax.random.split(
        position_pertubation_key, num=seen_events.shape[0]
    )
    seen_events = jax.vmap(perturb_midi_event_positions, (0, 0, None))(
        position_pertubation_keys,
        seen_events,
        float(duration_per_frame_in_secs),
    )

    # We can get rid of events that are BLANK_MIDI_EVENT for all samples in the batch
    # TODO: For now do not do this, as it leads to JAX recompilation pauses during the initial training steps
    # seen_events = seen_events[:, 0 : jnp.max(picked_midi_splits)]

    return {
        "audio_frames": selected_audio_frames,
        "seen_events": seen_events,
        "next_event": next_events,
        "duration_per_frame_in_secs": duration_per_frame_in_secs,
    }


class AudioToMidiDatasetLoader:
    SAMPLE_RATE = 44100.0

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
        self.queue = queue.Queue(prefetch_count + 1)
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

        self.loaded_midi_events, self.loaded_audio_frames, self.duration_per_frame = self._load_frames_from_disk(num_samples_to_load)
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
            yield self.queue.get()

    def _generate_batch(self, key: jax.random.PRNGKey):
        while not self._stop_event.is_set():
            with self.sample_load_lock:
                current_loaded_midi_events = self.loaded_midi_events
                current_loaded_audio_samples = self.loaded_audio_frames

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

            batch = generate_batch(
                batch_key,
                self.batch_size,
                self.duration_per_frame,
                selected_midi_events,
                selected_samples,
            )
            success = False
            while not success and not self._stop_event.is_set():
                try:
                    self.queue.put(batch, timeout=1)
                    success = True
                except queue.Full:
                    # This was not successful
                    pass


    def _load_frames_from_disk(self, num_samples_to_load: int, minimum_midi_event_size: Optional[int] = None):
        picked_samples = random.sample(self.all_sample_names, min(num_samples_to_load, len(self.all_sample_names)))

        loaded_audio_frames, sample_rate, duration_per_frame = AudioToMidiDatasetLoader.load_audio_frames(
            self.dataset_dir,
            picked_samples,
            sharding=self.sharding,
        )

        loaded_midi_events = (
            AudioToMidiDatasetLoader.load_midi_events_frame_time_positions(
                self.dataset_dir, picked_samples, duration_per_frame, minimum_size=minimum_midi_event_size
            )
        )

        return loaded_midi_events, loaded_audio_frames, duration_per_frame

    def _periodic_refresh_samples(
        self,
        num_samples_to_load: int,
        num_samples_to_maintain: int,
        sleep_time: int = 1 # seconds
    ):
        # TODO: Consider doing this in a way that preserves determinism
        while True:
            self._stop_event.wait(sleep_time)
            if self._stop_event.is_set():
                return

            with self.sample_load_lock:
                # Try to avoid unncessary JIT's due to different midi event lengths
                current_events = self.loaded_midi_events
                current_frames = self.loaded_audio_frames
            
            print(f"Loading {num_samples_to_load}, current size is {current_events.shape[0]}")
            loaded_midi_events, loaded_audio_frames, _duration_per_frame = self._load_frames_from_disk(num_samples_to_load, current_events.shape[1])

            current_amount_to_evict = max(0, num_samples_to_load - (num_samples_to_maintain - current_events.shape[0]))
            current_events = current_events[current_amount_to_evict:, ...]
            current_frames = current_frames[current_amount_to_evict:, ...]

            # I am intentionally using `np` here and not `jnp` so the maintained frames are only on the host
            blank_event = np.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=np.int16)
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

            with self.sample_load_lock:
                self.loaded_midi_events = new_events
                self.loaded_audio_frames = new_frames


    @classmethod
    def load_audio_frames(cls, dataset_dir: Path, sample_names: [str], sharding = None):
        audio_samples = parallel_audio_reader.load_audio_files(
            AudioToMidiDatasetLoader.SAMPLE_RATE,
            [dataset_dir / f"{sample_name}.aac" for sample_name in sample_names],
            MAX_EVENT_TIMESTAMP * 1000
        )
        if sharding is not None:
            audio_samples = jax.device_put(audio_samples, sharding)

        # TODO(knielsen): Consider factoring this out to some shared place
        desired_fft_duration = 20 # ms
        samples_per_fft = next_power_of_2(int(AudioToMidiDatasetLoader.SAMPLE_RATE * (desired_fft_duration / 1000)))
        duration_per_frame = samples_per_fft / AudioToMidiDatasetLoader.SAMPLE_RATE
        frames = jax.vmap(fft_audio, (0, None))(audio_samples, samples_per_fft)

        # Select only the lowest 10_000 Hz frequencies
        cutoff_frame = int(10_000 * duration_per_frame)
        frames = frames[:, 0:cutoff_frame, :]

        return jnp.transpose(frames, axes=(0, 2, 1)), AudioToMidiDatasetLoader.SAMPLE_RATE, duration_per_frame

    @classmethod
    def load_midi_events_real_time_positions(
        cls, dataset_dir: Path, sample_names: [str], minimum_size: Optional[int] = None
    ):
        unpadded_midi_events = list(
            map(lambda sample: events_from_sample(dataset_dir, sample), sample_names)
        )
        max_events_length = max([events.shape[0] for events in unpadded_midi_events])
        if minimum_size is not None:
            # We support extra padding to avoid JAX jit recompilations
            max_events_length = max(max_events_length, minimum_size)
        padding = np.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=np.int16)[
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
    def load_midi_events_frame_time_positions(
        cls, dataset_dir: Path, sample_names: [str], duration_per_frame: float, minimum_size: Optional[int] = None
    ):
        events_real_time_positions = (
            AudioToMidiDatasetLoader.load_midi_events_real_time_positions(
                dataset_dir, sample_names, minimum_size
            )
        )
        return event_positions_to_frame_time(
            events_real_time_positions, duration_per_frame
        )

    @classmethod
    def load_sample_names(cls, dataset_dir: Path):
        audio_names = set(
            map(lambda path: path[(len(str(dataset_dir)) + 1):-4], glob.glob(f"{dataset_dir}/**/*.aac", recursive=True))
        )
        label_names = set(
            map(lambda path: path[(len(str(dataset_dir)) + 1):-4], glob.glob(f"{dataset_dir}/**/*.csv", recursive=True))
        )

        if audio_names != label_names:
            raise "Did not find the same set of labels and samples!"

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
    duration_per_frame: float, frames: NDArray[jnp.float32]
):
    fig, ax1 = plt.subplots()

    # TODO(knielsen): Extract this so it can be used for training, as it is probably better signal?
    # transposed_reals = jnp.transpose(jnp.real(frames))
    # transformed_data = transposed_reals[0:int(transposed_reals.shape[0] / 2), :]
    X = jnp.linspace(0.0, duration_per_frame * frames.shape[0], frames.shape[0])
    Y = jnp.linspace(0.0, frames.shape[1] / duration_per_frame, frames.shape[1])
    ax1.pcolor(X, Y, jnp.transpose(frames))

    ax1.set(
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
        title="Audio signal in frequency-domain",
    )

    ax2 = ax1.twiny()
    ax2.set_xlim(0, frames.shape[0])
    ax2.set_xlabel("Frame count")


def visualize_sample(
    frames: Float[Array, "num_samples"],
    seen_events: Integer[Array, "seen_array_length 2"],
    next_event: Integer[Array, "2"],
    duration_per_frame_in_secs: float,
):
    print("Frames shape:", frames.shape)
    print("Duration per frame:", duration_per_frame_in_secs)
    print(f"Seen events: {seen_events}")
    print(f"Next event: {next_event}")
    plot_frequency_domain_audio(duration_per_frame_in_secs, frames)

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
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
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
        dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/v1"),
        batch_size=16,
        prefetch_count=0,
        num_workers=1,
        key=key,
        num_samples_to_load=5,
        num_samples_to_maintain=200,
    )
    dataset_loader_iter = iter(dataset_loader)

    for num, loaded_batch in zip(range(0, 100), dataset_loader_iter):
        print(f"Audio frames shape {num}:", loaded_batch["audio_frames"].shape)
        print(f"Seen events shape {num}:", loaded_batch["seen_events"].shape)
        print(f"Next event shape: {num}", loaded_batch["next_event"].shape)

        batch_idx = random.randint(0, loaded_batch["audio_frames"].shape[0] - 1)
        visualize_sample(
            loaded_batch["audio_frames"][batch_idx],
            loaded_batch["seen_events"][batch_idx],
            loaded_batch["next_event"][batch_idx],
            loaded_batch["duration_per_frame_in_secs"],
        )
