import argparse
import tensorflow as tf
from audio_to_midi_dataset import plot_output_probs
from infer import stitch_output_probs
import matplotlib.pyplot as plt
import rust_plugins
import math
import numpy as np
from audio_to_midi_dataset import MAX_EVENT_TIMESTAMP, AudioToMidiDatasetLoader

parser = argparse.ArgumentParser(description='infer_tf Example utility to show how to infer using TensorFlow instead of JAX.')
parser.add_argument('file', help='The path to the audio file to infer')
parser.add_argument('--overlap', type=float, default=0.5, help='The overlap value (default: 0.5)')

def create_audio_samples_window(overlap: float):
    window_size = round(MAX_EVENT_TIMESTAMP * AudioToMidiDatasetLoader.SAMPLE_RATE)
    overlap = round(overlap * AudioToMidiDatasetLoader.SAMPLE_RATE)

    step = window_size - overlap
    n_windows = math.ceil((audio_samples.shape[1] - overlap) / step)
    windows = []
    for i in range(n_windows):
        window_samples = audio_samples[:, i * step:i * step + window_size]
        # Make sure the window has the exact length (i.e. pad the last window if necessary)
        window_samples = np.pad(window_samples, ((0,0), (0, window_size - window_samples.shape[1])), constant_values=(0,0))
        windows.append(window_samples)
    return np.stack(windows)

model = tf.saved_model.load("./tf_export/")

args = parser.parse_args()
audio_file = args.file
overlap = args.overlap

audio_samples = rust_plugins.load_full_audio(str(audio_file), AudioToMidiDatasetLoader.SAMPLE_RATE)
windowed_samples = create_audio_samples_window(overlap)
frames, duration_per_frame, frame_width_in_secs = AudioToMidiDatasetLoader._convert_samples(windowed_samples)

_logits, probs = model.predict(frames)

stitched_probs = stitch_output_probs(probs, duration_per_frame, overlap)
plot_output_probs(audio_file, np.array(duration_per_frame), np.array(stitched_probs))

plt.show()
