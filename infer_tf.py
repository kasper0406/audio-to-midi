import tensorflow as tf
from audio_to_midi_dataset import AudioToMidiDatasetLoader, plot_output_probs
from infer import stitch_output_probs
import matplotlib.pyplot as plt

audio_file = "/Volumes/git/ml/datasets/midi-to-sound/validation_set_only_yamaha/C major scale.aif"
overlap = 0.5

frames, duration_per_frame, frame_width = AudioToMidiDatasetLoader.load_and_slice_full_audio(audio_file, overlap=overlap)

model = tf.saved_model.load("./tf_export/")
_logits, probs = model.predict(frames)

stitched_probs = stitch_output_probs(probs, duration_per_frame, overlap)
plot_output_probs(audio_file, duration_per_frame, stitched_probs)

plt.show()
