import argparse
from pathlib import Path
import jax
from infer import load_newest_checkpoint, batch_infer
from audio_to_midi_dataset import AudioToMidiDatasetLoader, plot_frequency_domain_audio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='audio_to_midi a utility to convert piano audio files to midi events.')
parser.add_argument('audio_file', help='The audio file to load and process')

parser.add_argument('--visualize',
    help='Visualize the audio samples and event probabilities using matplotlib',
    action='store_true')

args = parser.parse_args()

key = jax.random.PRNGKey(1234)

audio_file = Path(args.audio_file)
if not audio_file.exists():
    raise f"The specified audio file {audio_file} does not exist!"

frames, duration_per_frame, frame_width = AudioToMidiDatasetLoader.load_and_slice_full_audio(audio_file, overlap=0.5)
print("Loaded samples")
if args.visualize:
    for frame in frames:
        plot_frequency_domain_audio(str(audio_file), duration_per_frame, frame_width, frame)
    plt.show(block=True)

current_directory = Path(__file__).resolve().parent

checkpoint_path = current_directory / "audio_to_midi_checkpoints"
audio_to_midi = load_newest_checkpoint(checkpoint_path)

events, metadata = batch_infer(audio_to_midi, key, frames)
print(f"Inferred midi events: {events}")

plt.show() # Wait for matplotlib
