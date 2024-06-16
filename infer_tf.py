import argparse
import tensorflow as tf
from audio_to_midi_dataset import plot_output_probs
from infer import stitch_output_probs
import matplotlib.pyplot as plt
import modelutil
import math
import numpy as np
from audio_to_midi_dataset import MODEL_AUDIO_LENGTH, AudioToMidiDatasetLoader

parser = argparse.ArgumentParser(description='infer_tf Example utility to show how to infer using TensorFlow instead of JAX.')
parser.add_argument('file', help='The path to the audio file to infer')
parser.add_argument('--overlap', type=float, default=0.25, help='The overlap value (default: 0.25)')
# parser.add_argument('--tflite', help='If set, the TFLite model will be used used for inference', action='store_true')

args = parser.parse_args()
audio_file = args.file
overlap = args.overlap

sample_windows, window_duration = AudioToMidiDatasetLoader.load_and_slice_full_audio(audio_file, overlap=overlap)

#if not args.tflite:
model = tf.saved_model.load("./tf_export/")
all_probs = []
for window in sample_windows:
    _logits, probs = model.predict(window)
    all_probs.append(probs)
duration_per_frame = window_duration / all_probs[0].shape[0]
#else:
#    print("Creating tflite interpreter")
#    interpreter = tf.lite.Interpreter(model_path="model.tflite")
#    signatures = interpreter.get_signature_list()
#
#    predict = interpreter.get_signature_runner('predict')
#    output = predict(data=sample_windows)
#    probs = output["probs"]

print(f"Duration per frame {duration_per_frame}")
stitched_probs = stitch_output_probs(all_probs, duration_per_frame, overlap)
plot_output_probs(audio_file, np.array(duration_per_frame), np.array(stitched_probs))

plt.show()
