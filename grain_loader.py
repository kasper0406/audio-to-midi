from pathlib import Path
from typing import Optional, List

import grain.python as grain
import numpy as np
import pprint
import jax

from audio_to_midi_dataset import AudioToMidiDatasetLoader
import modelutil
from dataclasses import dataclass


# Wrapper around modelutil.DatasetTransformSettings to make it pickleable
@dataclass(frozen=True)
class TransformSettings:
    cut_probability: float
    rotate_probability: float
    random_erasing_probability: float
    mixup_probability: float
    gain_probability: float
    noise_probability: float
    label_smoothing_alpha: float

    def to_modelutils(self):
        return modelutil.DatasetTransfromSettings(
            cut_probability=self.cut_probability,
            rotate_probability=self.rotate_probability,
            random_erasing_probability=self.random_erasing_probability,
            mixup_probability=self.mixup_probability,
            gain_probability=self.gain_probability,
            noise_probability=self.noise_probability,
            label_smoothing_alpha=self.label_smoothing_alpha,
        )


class AudioToMidiSource(grain.RandomAccessDataSource):

    dataset_dir: Path
    all_sample_names: np.ndarray
    sample_rate: int
    audio_duration: float
    mini_batch_size: int
    output_divisions: int
    transform_settings: Optional[TransformSettings]

    def __init__(
            self,
            dataset_dir: Path,
            output_divisions: int,
            sample_rate: int,
            audio_duration: float,
            transform_settings: Optional[TransformSettings] = None,
            mini_batch_size: int = 32
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.mini_batch_size = mini_batch_size
        self.output_divisions = output_divisions
        self.all_sample_names = AudioToMidiDatasetLoader.load_sample_names(dataset_dir)
        self.transform_settings = transform_settings

    def __getitem__(self, idx):
        mini_batch_start_idx = idx * self.mini_batch_size
        sample_names_slice = slice(mini_batch_start_idx, min(len(self.all_sample_names), mini_batch_start_idx + self.mini_batch_size))
        samples_to_load = self.all_sample_names[sample_names_slice]

        if self.transform_settings:
            # print(f"Loading idx {idx} with transformations")
            midi_events, audio, sample_names = AudioToMidiDatasetLoader.load_samples_with_transformations(self.dataset_dir, self.output_divisions, samples_to_load, self.sample_rate, self.audio_duration, self.transform_settings.to_modelutils())
        else:
            # print(f"Loading idx {idx} without transformations")
            midi_events, audio, sample_names = AudioToMidiDatasetLoader.load_samples(self.dataset_dir, self.output_divisions, samples_to_load, self.sample_rate, self.audio_duration)

        # print(f"Audio for idx {idx} was loaded...")

        # Notice the result may be larger than the mini-batch size
        return midi_events.astype(np.float16), audio.astype(np.float16) # , sample_names

    def __len__(self):
        return int(len(self.all_sample_names) / self.mini_batch_size)


def extend_batch_fn(mini_batches):
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *mini_batches)


def create_dataset_loader(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    sample_rate: int = 16000,
    duration: float = 5.0,
    output_divisions: int = 50,
    transform_settings: Optional[TransformSettings] = None,
    seed: int = 42,
) -> grain.IterDataset:
    mini_batch_size = 16
    source = AudioToMidiSource(
        dataset_dir,
        output_divisions=output_divisions,
        sample_rate=sample_rate,
        audio_duration=duration,
        transform_settings=transform_settings,
        mini_batch_size=mini_batch_size,
    )
    dataset = (
        grain.MapDataset.source(source)
            .seed(seed)
            .repeat(num_epochs)
            .shuffle()
            .batch(batch_size=max(1, int(batch_size / mini_batch_size)), batch_fn=extend_batch_fn)
    )

    iter_dataset = dataset.to_iter_dataset(grain.ReadOptions(
        num_threads=2,
        prefetch_buffer_size=10,
    )).prefetch(grain.MultiprocessingOptions(
        num_workers=num_workers,
        # enable_profiling=True,
    ))

    return iter(iter_dataset)


if __name__ == "__main__":
    batch_size = 32
    batches_to_load = 100
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/logic/logic_dataset_2")

    iter_dataset = create_dataset_loader(dataset_dir, batch_size, num_workers=4, num_epochs=1)

    for element, batch_idx in zip(iter_dataset, range(batches_to_load)):
        events, audio = element
        print("Events:")
        pprint.pprint(events)
        print("Audio:")
        pprint.pprint(audio)
