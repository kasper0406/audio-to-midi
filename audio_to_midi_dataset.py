import csv
import glob
import queue
import random
import threading
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from numpy.typing import NDArray
from pydub import AudioSegment


def load_audio_and_normalize(file: str) -> (int, NDArray[jnp.float32]):
    """Loads an audio file and returns the sample rate along with the normalized samples."""
    SAMPLE_RATE = 44100.0
    audio = AudioSegment.from_file(file, "aac")
    resampled_audio = audio.set_frame_rate(
        SAMPLE_RATE
    )  # Resample to the frequency we operate in

    left_channel_samples = resampled_audio.split_to_mono()[0].get_array_of_samples()
    sample_array = jnp.array(left_channel_samples).T.astype(jnp.float32)
    normalized_samples = sample_array / jnp.max(jnp.abs(sample_array))

    return SAMPLE_RATE, normalized_samples


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


@partial(jax.jit, static_argnames=["sample_rate", "fft_duration"])
def fft_audio(
    sample_rate: int, samples: NDArray[jnp.float32], fft_duration=20
) -> (Float, NDArray[jnp.float32]):
    """Computes the fft of the audio samples
    Returns a tuple of the frame duration in seconds, and the complex FFT components

    Args:
        fft_duration: The duration in ms to compute the fft for. It will be rounded to the next
                      power of 2 samples
    """
    samples_per_fft = next_power_of_2(int(sample_rate * (fft_duration / 1000)))

    num_padding_symbols = samples_per_fft - (samples.shape[0] % samples_per_fft)
    if num_padding_symbols == samples_per_fft:
        num_padding_symbols = 0
    padded_data = jnp.pad(samples, (0, num_padding_symbols), constant_values=0)
    data = padded_data.reshape(
        (int(padded_data.shape[0] / samples_per_fft), samples_per_fft)
    )

    fft = jnp.fft.fft(data)
    # features = jnp.reshape(jnp.stack((fft.real, fft.imag), axis=2), (fft.shape[0], 2 * fft.shape[1]))
    # return features
    return float(samples_per_fft / sample_rate), fft


@partial(jax.jit, static_argnames=["duration_per_frame", "high_freq_cutoff"])
def cleanup_fft_and_low_pass(
    frames: Float[Array, "frame_count frame_size"],
    duration_per_frame: Float,
    high_freq_cutoff: Float = 10000,
):
    # Drop phase information and drop high-frequencies above the `high_freq_cut_off` wavelength
    transposed_reals = jnp.transpose(jnp.real(frames))
    frequencies_to_include = int(high_freq_cutoff * duration_per_frame)
    processed_frequencies = transposed_reals[0:frequencies_to_include, :]
    return processed_frequencies, duration_per_frame


def load_sample_names(dataset_dir: str):
    audio_names = set(
        map(lambda path: Path(path).stem, glob.glob(f"{dataset_dir}/*.aac"))
    )
    label_names = set(
        map(lambda path: Path(path).stem, glob.glob(f"{dataset_dir}/*.csv"))
    )

    if audio_names != label_names:
        raise "Did not find the same set of labels and samples!"

    return list(audio_names)


def audio_features_from_sample(dataset_dir: str, sample: str, key: jax.random.PRNGKey):
    sample_rate, samples = load_audio_and_normalize(f"{dataset_dir}/{sample}.aac")
    perturbed_samples = perturb_audio_sample(samples, key)

    duration_per_frame, frequency_domain = fft_audio(sample_rate, perturbed_samples)
    frames, duration_per_frame = cleanup_fft_and_low_pass(
        frequency_domain, float(duration_per_frame)
    )
    return frames, sample_rate, duration_per_frame


def events_from_sample(dataset_dir: str, sample: str) -> [(float, int)]:
    events = []
    max_event_timestamp = 0.0

    SEQUENCE_START = 0
    SEQUENCE_END = 1
    events.append((0.0, SEQUENCE_START))

    with open(f"{dataset_dir}/{sample}.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            attack_time = float(row[0])
            duration = float(row[1])
            key = int(row[2])
            velocity = float(row[3])

            # TODO: Make the event calculation nicer using the Midi dictionary
            # Notice we put releases prior to attacks in terms of midi event index
            events.append((attack_time, 2 + 88 + (key - 55)))  # Attack
            events.append((attack_time + duration, 2 + (key - 55)))  # Release

            max_event_timestamp = max(max_event_timestamp, attack_time + duration)

    events.append((max_event_timestamp, SEQUENCE_END))

    return sorted(events)


def audio_to_midi_dataset_generator(
    key: jax.random.PRNGKey,
    batch_size: int = 128,
    dataset_dir: str = "/Volumes/git/ml/datasets/midi-to-sound/v0",
):
    samples = load_sample_names(dataset_dir)
    midi_events_by_sample = list(
        map(lambda sample: events_from_sample(dataset_dir, sample), samples)
    )

    while True:
        batch_frames = []
        batch_seen_events = []
        batch_next_event = []

        while len(batch_frames) < batch_size:
            key, sample_key = jax.random.split(key, num=2)

            sample_index = random.randint(0, len(samples) - 1)
            sample = samples[sample_index]
            midi_events = midi_events_by_sample[sample_index]

            (
                audio_frames,
                sample_rate,
                duration_per_frame_in_secs,
            ) = audio_features_from_sample(dataset_dir, sample, sample_key)

            def position_to_frame_index(position_in_seconds: float) -> int:
                return int(position_in_seconds / (duration_per_frame_in_secs))

            # Pick a split inside the events to predict
            midi_event_split = random.randint(1, len(midi_events) - 1)
            seen_midi_events = midi_events[:midi_event_split]
            next_event_to_predict = midi_events[midi_event_split]

            batch_frames.append(jnp.transpose(audio_frames))
            # Tuple of (midi event nr, position in frame index)
            batch_seen_events.append(
                [
                    (event[1], position_to_frame_index(event[0]))
                    for event in seen_midi_events
                ]
            )
            batch_next_event.append(
                (
                    next_event_to_predict[1],
                    position_to_frame_index(next_event_to_predict[0]),
                )
            )

        # TODO: Make this nicer
        max_seen_events_length = max(len(sublist) for sublist in batch_seen_events)
        padded_batch_seen_events = [
            sublist + [(-1, 0)] * (max_seen_events_length - len(sublist))
            for sublist in batch_seen_events
        ]

        yield {
            "audio_frames": jnp.stack(batch_frames, axis=0),
            "seen_events": jnp.array(padded_batch_seen_events),
            "next_event": jnp.array(batch_next_event),
        }


class AudioToMidiDatasetLoader:
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        prefetch_count: int,
        num_workers: int,
        key: jax.random.PRNGKey,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.queue = queue.Queue(prefetch_count + 1)

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
            key, self.batch_size, self.dataset_dir
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
    fig, ax = plt.subplots()

    # TODO(knielsen): Extract this so it can be used for training, as it is probably better signal?
    # transposed_reals = jnp.transpose(jnp.real(frames))
    # transformed_data = transposed_reals[0:int(transposed_reals.shape[0] / 2), :]
    X = jnp.linspace(0.0, duration_per_frame * frames.shape[1], frames.shape[1])
    Y = jnp.linspace(0.0, frames.shape[0] / duration_per_frame, frames.shape[0])
    ax.pcolor(X, Y, frames)

    ax.set(
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
        title="Audio signal in frequency-domain",
    )
    ax.grid()


if __name__ == "__main__":
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

    dataset_loader = AudioToMidiDatasetLoader(
        dataset_dir="/Volumes/git/ml/datasets/midi-to-sound/v0",
        batch_size=64,
        prefetch_count=20,
        num_workers=4,
        key=key,
    )
    dataset_loader_iter = iter(dataset_loader)

    loaded_batch = next(dataset_loader_iter)
    print("Audio frames shape:", loaded_batch["audio_frames"].shape)
    print("Seen events shape:", loaded_batch["seen_events"].shape)
    print("Next event shape:", loaded_batch["next_event"].shape)
