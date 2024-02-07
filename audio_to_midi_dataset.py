import csv
import glob
import queue
import random
import threading
from functools import partial
from pathlib import Path

import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import jax.sharding as sharding
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, Integer
from numpy.typing import NDArray

import parallel_audio_reader


@jax.jit
def normalize_audio(
    samples: Float[Array, "num_samples"],
) -> Float[Array, "num_samples"]:
    sample_array = jnp.array(samples).T.astype(jnp.float32)
    normalized_samples = sample_array / jnp.max(jnp.abs(sample_array))
    return normalized_samples


@jax.jit
def perturb_audio_sample(
    samples, key: jax.random.PRNGKey
) -> (int, NDArray[jnp.float32]):
    """In order to make overfitting less likely this function perturbs the audio sampel in various ways:
    1. Add gausian noise
    """
    sigma = jax.random.uniform(key) / 10  # Randomize the level of noise
    gaussian_noise = sigma * jax.random.normal(key, samples.shape)
    return jax.numpy.clip(samples + gaussian_noise, -1.0, 1.0)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@partial(jax.jit, static_argnames=["sample_rate", "desired_fft_duration"])
def fft_audio(
    sample_rate: int, samples: NDArray[jnp.float32], desired_fft_duration=20
) -> (Float, NDArray[jnp.float32]):
    """Computes the fft of the audio samples
    Returns a tuple of the frame duration in seconds, and the complex FFT components

    Args:
        fft_duration: The duration in ms to compute the fft for. It will be rounded to the next
                      power of 2 samples
    """
    samples_per_fft = next_power_of_2(int(sample_rate * (desired_fft_duration / 1000)))

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
    compression_factor = 100
    compressed_amplitudes = (
        jnp.sign(transposed_amplitudes)
        * jnp.log1p(compression_factor * jnp.abs(transposed_amplitudes))
        / jnp.log1p(compression_factor)
    )

    # Normalize the coefficients to give them 0 mean and unit variance
    standardized_amplitudes = (
        compressed_amplitudes - jnp.mean(compressed_amplitudes)
    ) / jnp.var(compressed_amplitudes)

    return jnp.float32(samples_per_fft / sample_rate), standardized_amplitudes


def load_sample_names(dataset_dir: str):
    audio_names = set(
        map(lambda path: Path(path).stem, glob.glob(f"{dataset_dir}/*.aac"))
    )
    label_names = set(
        map(lambda path: Path(path).stem, glob.glob(f"{dataset_dir}/*.csv"))
    )

    if audio_names != label_names:
        raise "Did not find the same set of labels and samples!"

    # Test, for now just return the first sample
    return list(audio_names)


@partial(jax.jit, static_argnames=["sample_rate", "fixed_duration_in_seconds"])
def pad_or_trim(
    sample_rate: float,
    samples: Float[Array, "num_samples"],
    fixed_duration_in_seconds: float = 5,
):
    """Trim the audio to exactly 5 seconds in duration. Padding with empty audio if shorter, truncating if longer"""
    desired_sample_count = int(sample_rate * fixed_duration_in_seconds)
    return jnp.pad(samples, (0, max(0, desired_sample_count - samples.shape[0])))[
        0:desired_sample_count
    ]


@partial(jax.jit, static_argnames=["sample_rate"])
def audio_features_from_sample(
    sample_rate: float, samples: Float[Array, "num_samples"], key: jax.random.PRNGKey
):
    padded_and_trimmed_samples = pad_or_trim(sample_rate, samples)
    perturbed_samples = perturb_audio_sample(padded_and_trimmed_samples, key)
    duration_per_frame, frequency_domain = fft_audio(sample_rate, perturbed_samples)

    return (
        frequency_domain,
        sample_rate,
        duration_per_frame,
    )


def events_from_sample(
    dataset_dir: str, sample: str
) -> Float[Array, "max_number_of_events 2"]:
    """
    Returns a numpy array with tuples of (timestamp in seconds, midi event id)
    """
    max_event_timestamp = 0.0

    SEQUENCE_START = 0
    SEQUENCE_END = 1

    events = []
    with open(f"{dataset_dir}/{sample}.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            attack_time = float(row[0])
            duration = float(row[1])
            key = int(row[2])
            velocity = float(row[3])

            # TODO: Make the event calculation nicer using the Midi dictionary
            # Notice we put releases prior to attacks in terms of midi event index
            events.append((attack_time, 2 + 88 + (key - 21)))  # Attack
            events.append((attack_time + duration, 2 + (key - 21)))  # Release

            max_event_timestamp = max(max_event_timestamp, attack_time + duration)

    # Append the SEQUENCE_START and SEQUENCE_EVENT events outside the sorting
    # TODO: Find a nicer way...
    # TODO: Should the max sequence event timestamp here instead be the length of the audio sequence!?
    return jnp.array(
        [(0.0, SEQUENCE_START)] + sorted(events) + [(max_event_timestamp, SEQUENCE_END)]
    )


def audio_to_midi_dataset_generator(
    key: jax.random.PRNGKey,
    batch_size: int,
    sample_rate: float,
    all_midi_events: list[(int, int)],
    all_audio_samples: Float[Array, "num_files samples"],
):
    assert (
        len(all_midi_events) == all_audio_samples.shape[0]
    ), "The number of loaded files should be equal!"
    file_count = len(all_midi_events)

    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    shard = sharding.PositionalSharding(devices)

    while True:
        key, indexes_key, sample_key, midi_split_key = jax.random.split(key, num=4)
        indexes = jax.random.randint(
            indexes_key,
            shape=(batch_size,),
            minval=0,
            maxval=file_count,
            dtype=jnp.int32,
        )
        sample_keys = jax.random.split(sample_key, num=batch_size)
        selected_samples = jax.device_put(all_audio_samples[indexes], shard)

        selected_audio_frames, _, duration_per_frame_in_secs = jax.vmap(
            audio_features_from_sample, in_axes=(None, 0, 0), out_axes=(0, None, None)
        )(sample_rate, selected_samples, sample_keys)

        # Select only the lowest 10_000 Hz frequencies
        cutoff_frame = int(10_000 * duration_per_frame_in_secs)
        selected_audio_frames = selected_audio_frames[:, 0:cutoff_frame, :]

        @jax.jit
        def position_to_frame_index(event: Integer[Array, "2"]) -> Integer[Array, "2"]:
            # TODO: There is an ugly swap of position and time here! I should fix this
            return jnp.array(
                [event[1], jnp.int32(event[0] / duration_per_frame_in_secs)],
                dtype=jnp.int32,
            )

        selected_midi_events = all_midi_events[indexes]
        # Express midi event position in terms of frames
        # TODO: Consider doing this outside the loop
        selected_midi_events = jnp.apply_along_axis(
            position_to_frame_index, 2, selected_midi_events
        )

        midi_event_counts = jnp.count_nonzero(
            selected_midi_events != -1, axis=(1)
        )[
            :, 0
        ]  # Count along the event id dimension and exclude any padded events with value -1, and pick any split pointing to the index to split _before_
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
        seen_events = selected_midi_events.at[~seen_event_mask].set(jnp.array([-1, 0]))

        # We can get rid of events that are -1 for all samples in the batch
        # TODO: For now do not do this, as it leads to JAX recompilation pauses during the initial training steps
        # seen_events = seen_events[:, 0 : jnp.max(picked_midi_splits)]

        yield {
            "audio_frames": jnp.transpose(selected_audio_frames, axes=(0, 2, 1)),
            "seen_events": seen_events,
            "next_event": next_events,
            "duration_per_frame_in_secs": duration_per_frame_in_secs,
        }


class AudioToMidiDatasetLoader:
    def __init__(
        self,
        dataset_dir: Path,
        batch_size: int,
        prefetch_count: int,
        num_workers: int,
        key: jax.random.PRNGKey,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.queue = queue.Queue(prefetch_count + 1)

        all_sample_names = load_sample_names(dataset_dir)

        unpadded_midi_events = list(
            map(
                lambda sample: events_from_sample(dataset_dir, sample), all_sample_names
            )
        )
        max_events_length = max([events.shape[0] for events in unpadded_midi_events])
        padded_midi_events = [
            jnp.pad(
                events,
                ((0, max_events_length - events.shape[0])),
                "constant",
                constant_values=(-1, 0),
            )
            for events in unpadded_midi_events
        ]
        self.all_midi_events = jnp.stack(padded_midi_events, axis=0)

        self.sample_rate = 44100.0
        all_audio_samples = jnp.array(
            parallel_audio_reader.load_audio_files(
                self.sample_rate,
                [dataset_dir / f"{name}.aac" for name in all_sample_names],
            )
        )
        self.all_audio_samples = jax.vmap(normalize_audio)(all_audio_samples)

        worker_keys = jax.random.split(key, num=num_workers)
        for worker_id in range(num_workers):
            threading.Thread(
                target=partial(self._generate_batch, key=worker_keys[worker_id]),
                daemon=True,
            ).start()

    def __iter__(self):
        while True:
            yield self.queue.get()

    def _generate_batch(self, key: jax.random.PRNGKey):
        batch_generator = audio_to_midi_dataset_generator(
            key,
            self.batch_size,
            self.sample_rate,
            self.all_midi_events,
            self.all_audio_samples,
        )
        while True:
            self.queue.put(next(batch_generator))


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


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    key = jax.random.PRNGKey(4)

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
        dataset_dir=Path("/Volumes/git/ml/datasets/midi-to-sound/v0"),
        batch_size=128,
        prefetch_count=20,
        num_workers=4,
        key=key,
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
