import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from jaxtyping import Array, Float
from pydub import AudioSegment


def load_audio(file: str, sample_rate) -> Float[Array, "num_samples"]:
    """Loads an audio file and returns the sample rate along with the normalized samples."""
    audio = AudioSegment.from_file(file, "aac")
    resampled_audio = audio.set_frame_rate(
        sample_rate
    )  # Resample to the frequency we operate in

    left_channel_samples = resampled_audio.split_to_mono()[0].get_array_of_samples()
    return left_channel_samples


def load_audio_files(
    sample_rate, files_to_load: list[Path]
) -> list[Float[Array, "num_samples"]]:
    with ThreadPoolExecutor() as executor:
        all_audio_frames = list(
            executor.map(
                partial(load_audio, sample_rate=sample_rate),
                files_to_load,
            )
        )
    return all_audio_frames


if __name__ == "__main__":
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")
    sample_rate = 44100.0

    print("Loading audio files")
    audio_files = list(map(lambda path: Path(path), glob.glob(f"{dataset_dir}/*.aac")))
    load_audio_files(sample_rate, audio_files)

    print("Done loading audio files")
