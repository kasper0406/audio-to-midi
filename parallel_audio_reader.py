import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import numpy as np

from typing import Optional
from jaxtyping import Array, Float
from pydub import AudioSegment


def load_audio(file: Path, sample_rate: float, duration: Optional[int] = 5000, pad_to_fixed_duration: bool = True) -> Float[Array, "num_samples"]:
    """Loads an audio file and returns the sample rate along with the normalized samples."""
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(
        int(sample_rate)
    )  # Resample to the frequency we operate in

    # Make the audio be of length exactly `duration`
    if duration is not None:
        audio = audio[:duration] # Pick only the first `duration` ms
        padding_needed = duration - len(audio)
        if pad_to_fixed_duration and padding_needed > 0:
            silence = AudioSegment.silent(duration=padding_needed)
            audio = audio + silence

    all_channels = [ np.array(channel.get_array_of_samples()).T.astype(np.float16) for channel in audio.split_to_mono() ]
    all_channels = np.stack(all_channels)
    combined_channels = np.sum(all_channels, axis=0)

    max_sample_value = np.max(np.abs(combined_channels))
    if max_sample_value > 0.0: # Protect against silent audio files
        combined_channels = combined_channels / max_sample_value
    return combined_channels


def load_audio_files(
    sample_rate: float, files_to_load: list[Path], duration: 5000
) -> Float[Array, "num_files num_samples"]:
    with ThreadPoolExecutor(max_workers=64) as executor:
        all_audio_samples = list(
            executor.map(
                partial(load_audio, sample_rate=sample_rate, duration=duration),
                files_to_load,
            )
        )
    max_len = max([ samples.shape[0] for samples in all_audio_samples ])
    samples = [ np.pad(frame, (0, max_len - len(frame)), 'constant', constant_values=(0)) for frame in all_audio_samples ]
    return np.stack(samples)


if __name__ == "__main__":
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")
    sample_rate = 44100.0

    print("Loading audio files")
    audio_files = list(map(lambda path: Path(path), glob.glob(f"{dataset_dir}/*.aac")))
    load_audio_files(sample_rate, audio_files)

    print("Done loading audio files")
