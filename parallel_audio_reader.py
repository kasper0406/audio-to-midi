import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import numpy as np

from jaxtyping import Array, Float
from pydub import AudioSegment


def load_audio(file: str, sample_rate, duration: 5000) -> Float[Array, "num_samples"]:
    """Loads an audio file and returns the sample rate along with the normalized samples."""
    audio = AudioSegment.from_file(file, "aac")
    audio = audio.set_frame_rate(
        sample_rate
    )  # Resample to the frequency we operate in

    # Make the audio be of length exactly `duration`
    audio = audio[:duration] # Pick only the first `duration` ms
    padding_needed = duration - len(audio)
    if padding_needed > 0:
        silence = AudioSegment.silent(duration=padding_needed)
        audio = audio + silence

    left_channel_samples = audio.split_to_mono()[0].get_array_of_samples()
    return left_channel_samples


def load_audio_files(
    sample_rate, files_to_load: list[Path], duration: 5000
) -> list[Float[Array, "num_samples"]]:
    with ThreadPoolExecutor() as executor:
        all_audio_frames = list(
            executor.map(
                partial(load_audio, sample_rate=sample_rate, duration=duration),
                files_to_load,
            )
        )
    max_len = max([ len(frame) for frame in all_audio_frames ])
    return [ np.pad(frame, (0, max_len - len(frame)), 'constant', constant_values=(0)) for frame in all_audio_frames ]


if __name__ == "__main__":
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")
    sample_rate = 44100.0

    print("Loading audio files")
    audio_files = list(map(lambda path: Path(path), glob.glob(f"{dataset_dir}/*.aac")))
    load_audio_files(sample_rate, audio_files)

    print("Done loading audio files")
