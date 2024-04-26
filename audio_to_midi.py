import argparse
from pathlib import Path
import jax
from infer import load_newest_checkpoint, batch_infer
from train import compute_testset_loss
from audio_to_midi_dataset import AudioToMidiDatasetLoader, plot_frequency_domain_audio, plot_embedding
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='audio_to_midi a utility to convert piano audio files to midi events.')
parser.add_argument('path', help='The path to the audio file or directory for validation')

parser.add_argument('--visualize-audio',
    help='Visualize the audio samples and event probabilities using matplotlib',
    action='store_true')
parser.add_argument('--visualize-audio-embeddings',
    help='Visualize the audio embeddings',
    action='store_true')
parser.add_argument('--validation',
    help='Test the provided validation set on the model',
    action='store_true')

args = parser.parse_args()

key = jax.random.PRNGKey(1234)
current_directory = Path(__file__).resolve().parent

checkpoint_path = current_directory / "audio_to_midi_checkpoints"
audio_to_midi = load_newest_checkpoint(checkpoint_path)

if not args.validation:
    audio_file = Path(args.path)
    if not audio_file.exists():
        raise f"The specified audio file {audio_file} does not exist!"

    frames, duration_per_frame, frame_width = AudioToMidiDatasetLoader.load_and_slice_full_audio(audio_file, overlap=0.5)
    print("Loaded samples")
    if args.visualize_audio:
        for frame in frames:
            plot_frequency_domain_audio(str(audio_file), duration_per_frame, frame_width, frame)
        plt.show(block=False)

    if args.visualize_audio_embeddings:
        # import code
        # code.interact(local=locals())
        for frame in frames:
            embeddings = audio_to_midi.frame_embedding(frame)
            print(f"Embeddings shape: {embeddings.shape}")
            plot_embedding(str(audio_file), embeddings)
        plt.show(block=False)

    events, metadata = batch_infer(audio_to_midi, key, frames)
    print(f"Inferred midi events: {events}")

if args.validation:
    validation_dir = Path(args.path)
    loss, idv_losses = compute_testset_loss(audio_to_midi, validation_dir, key, sharding=None)
    print(f"Validation loss: {loss}, idv = {idv_losses}")

plt.show() # Wait for matplotlib
