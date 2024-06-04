import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from infer import load_newest_checkpoint, predict_and_stitch, write_midi_file
from train import compute_testset_loss, compute_testset_loss_individual, compute_model_output_frames
from audio_to_midi_dataset import AudioToMidiDatasetLoader, plot_frequency_domain_audio, plot_embedding, visualize_sample, plot_output_probs
import matplotlib.pyplot as plt
import modelutil

parser = argparse.ArgumentParser(description='audio_to_midi a utility to convert piano audio files to midi events.')
parser.add_argument('path', help='The path to the audio file or directory for validation')
parser.add_argument('output', help='The output MIDI file', nargs='?')

parser.add_argument('--visualize-audio',
    help='Visualize the audio samples and event probabilities using matplotlib',
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
audio_to_midi, state = load_newest_checkpoint(checkpoint_path)

if not args.validation:
    audio_file = Path(args.path)
    if not audio_file.exists():
        raise f"The specified audio file {audio_file} does not exist!"

    overlap = 0.0
    sample_windows, window_duration = AudioToMidiDatasetLoader.load_and_slice_full_audio(audio_file, overlap=overlap)
    print("Loaded samples")
    
    individual_probs, stitched_probs, duration_per_frame = predict_and_stitch(audio_to_midi, state, sample_windows, window_duration, overlap=overlap)

    if args.visualize_audio:
        for i in range(individual_probs.shape[0]):
            visualize_sample(args.path, sample_windows[i], individual_probs[i])
            plt.show(block=False)
    
    print(f"Stitched probs shape: {stitched_probs.shape}")
    plot_output_probs(args.path, duration_per_frame, stitched_probs)
    plt.show(block=False)

    events = modelutil.extract_events(stitched_probs)
    if args.output:
        print(f"Writing MIDI file to {args.output}")
        write_midi_file(events, duration_per_frame, args.output)


if args.validation:
    validation_dir = Path(args.path)

    num_model_output_frames = compute_model_output_frames(audio_to_midi, state)
    if args.individual:
        losses = compute_testset_loss_individual(audio_to_midi, state, validation_dir, num_model_output_frames, key, sharding=None)
        for sample_name, losses in losses.items():
            loss = losses["loss"]
            hit_rate = losses["hit_rate"]
            eventized_diff = losses["eventized_diff"]
            phantom_note_diff = losses["phantom_note_diff"]
            missed_note_diff = losses["missed_note_diff"]
            print(f"{sample_name}\t{loss}\t{hit_rate}\t{eventized_diff}\t{phantom_note_diff}\t{missed_note_diff}")
    else:
        loss, hit_rate, eventized_diff = compute_testset_loss(audio_to_midi, state, validation_dir, num_model_output_frames, key, sharding=None)
        print(f"Validation loss: {loss}")
        print(f"Hit rate: {hit_rate}")
        print(f"Eventized diff: {eventized_diff}")

plt.show() # Wait for matplotlib
