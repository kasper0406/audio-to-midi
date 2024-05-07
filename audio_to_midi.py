import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from infer import load_newest_checkpoint, forward, write_midi_file
from train import compute_testset_loss, compute_testset_loss_individual
from audio_to_midi_dataset import AudioToMidiDatasetLoader, plot_frequency_domain_audio, plot_embedding, visualize_sample, plot_output_probs
import matplotlib.pyplot as plt
import rust_plugins

parser = argparse.ArgumentParser(description='audio_to_midi a utility to convert piano audio files to midi events.')
parser.add_argument('path', help='The path to the audio file or directory for validation')
parser.add_argument('output', help='The output MIDI file', nargs='?')

parser.add_argument('--visualize-audio',
    help='Visualize the audio samples and event probabilities using matplotlib',
    action='store_true')
parser.add_argument('--visualize-audio-embeddings',
    help='Visualize the audio embeddings',
    action='store_true')
parser.add_argument('--validation',
    help='Test the provided validation set on the model',
    action='store_true')
parser.add_argument('--individual',
    help='Report losses on individual samples in the validation set',
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

    overlap = 0.5
    frames, duration_per_frame, frame_width = AudioToMidiDatasetLoader.load_and_slice_full_audio(audio_file, overlap=overlap)
    print("Loaded samples")

    if args.visualize_audio_embeddings:
        # import code
        # code.interact(local=locals())
        for frame in frames:
            embeddings = audio_to_midi.frame_embedding(frame)
            print(f"Embeddings shape: {embeddings.shape}")
            plot_embedding(str(audio_file), embeddings)
        plt.show(block=False)
    
    individual_probs, stitched_probs = forward(audio_to_midi, frames, key, duration_per_frame, overlap=overlap)

    if args.visualize_audio:
        for i in range(individual_probs.shape[0]):
            # TODO: Fix this hack!
            padded_output = jnp.pad(individual_probs[i], ((0, frames[i].shape[1] - individual_probs[i].shape[0]), (0, 0)), 'constant', constant_values=0)
            visualize_sample(args.path, frames[i], padded_output, None, duration_per_frame, frame_width)
            plt.show(block=False)
    
    print(f"Stitched probs shape: {stitched_probs.shape}")
    plot_output_probs(args.path, duration_per_frame, stitched_probs)
    plt.show(block=False)

    events = rust_plugins.extract_events(stitched_probs)
    if args.output:
        print(f"Writing MIDI file to {args.output}")
        write_midi_file(events, duration_per_frame, args.output)


if args.validation:
    validation_dir = Path(args.path)

    if args.individual:
        losses = compute_testset_loss_individual(audio_to_midi, validation_dir, key, sharding=None)
        for sample_name, losses in losses.items():
            loss = losses["loss"]
            print(f"{sample_name}\t{loss}")
    else:
        loss = compute_testset_loss(audio_to_midi, validation_dir, key, sharding=None)
        print(f"Validation loss: {loss}")

plt.show() # Wait for matplotlib
