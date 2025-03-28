from pathlib import Path

import os
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import matplotlib.figure
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from dataclasses import dataclass
import sys
import argparse
import matplotlib
import optax

from model import OutputSequenceGenerator, model_config, get_model_metadata
from audio_to_midi_dataset import NUM_VELOCITY_CATEGORIES, MIDI_EVENT_VOCCAB_SIZE, plot_output_probs, AudioToMidiDatasetLoader, MODEL_AUDIO_LENGTH
import modelutil
from rope import precompute_frequencies, RopeFreqs

def change_fp_precision(model, dtype):
    def to_dtype(leaf):
        if eqx.is_inexact_array(leaf):
            return leaf.astype(dtype)
        return leaf
    return jax.tree_util.tree_map(to_dtype, model)

def stitch_output_probs(all_probs, duration_per_frame: float, overlap: float):
    return modelutil.stitch_probs(np.stack(all_probs), overlap, duration_per_frame)

def predict_and_stitch(model, state, samples, window_duration: float, overlap=0.0):
    rope_freqs = precompute_frequencies(model_config["attention_size"], 300)
    # samples = samples.astype(np.float16)
    _logits, probs = jax.vmap(model.predict, in_axes=(None, 0, None))(state, samples, rope_freqs)
    probs = probs.astype(np.float32)  # Convert probs to f32 to make them compatible with the rust plugin
    duration_per_frame = window_duration / probs.shape[1]
    print(f"Duration per frame: {duration_per_frame}")
    return probs, stitch_output_probs(probs, duration_per_frame, overlap), duration_per_frame

def write_midi_file(events: [(int, int, int, int)], duration_per_frame: float, output_file: str):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # TODO: Make a model that givne events predicts the time signature and tempo
    #       For now we'll just denote everything as 4/4 in tempo 120.
    tempo = 120
    time_signature = (4, 4)

    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    track.append(MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1], clocks_per_click=24, notated_32nd_notes_per_beat=8))

    def frame_to_midi_time(frame: int):
        seconds = frame * duration_per_frame
        ticks_per_beat = midi.ticks_per_beat
        microseconds_per_quarter_note = mido.bpm2tempo(tempo, time_signature)
        return mido.second2tick(seconds, ticks_per_beat, microseconds_per_quarter_note)

    # The actual midi events need to be specified with delta times, so first build the midi events
    # out of order, then we will sort and emit delta times
    out_of_order_midi_events = []
    for attack_frame, key, duration_frame, velocity in events:
        midi_time_attack = frame_to_midi_time(attack_frame)
        midi_time_release = frame_to_midi_time(attack_frame + duration_frame)
        midi_key = key + 21
        midi_velocity = int(round((velocity / NUM_VELOCITY_CATEGORIES) * 127))

        out_of_order_midi_events.append((midi_time_attack, 'note_on', midi_key, midi_velocity))
        out_of_order_midi_events.append((midi_time_release, 'note_off', midi_key, midi_velocity))

    current_time = 0
    for time, event_type, key, velocity in sorted(out_of_order_midi_events):
        delta_time = time - current_time
        track.append(Message(event_type, note=key, velocity=velocity, time=delta_time))
        current_time = time

    midi.save(output_file)

@dataclass
class DetailedEventLoss:
    full_diff: int
    phantom_notes_diff: float
    missed_notes_diff: float
    notes_hit: int
    hit_rate: float
    visualization: matplotlib.figure.Figure | None = None

def detailed_event_loss(
    output_probs: Float[Array, "seq_len midi_voccab_size"],
    expected: Float[Array, "seq_len midi_voccab_size"],
    generate_visualization: bool = False
) -> DetailedEventLoss:
    """
    Given predicted and expected fram events, compute a more detailed loss,
    to help evaluate how good a prediction is in a more detailed and "closer
    to the music way" than the ordinary loss function.
    """
    predicted = modelutil.extract_events(np.array(output_probs))
    predicted = modelutil.to_frame_events([predicted], output_probs.shape[0])[0]
    expected = expected[:predicted.shape[0]]

    full_diff = np.sum(np.abs(predicted - expected))

    played_predicted = predicted > 0
    played_expected = expected > 0

    phantom_notes_diff = np.sum(played_predicted & ~played_expected)
    # Note we discount the missed notes weighed by its severity (if the node is basically decayed it doesn't make much of a difference)
    missed_notes_diff = np.sum(expected[played_expected & ~played_predicted])
    notes_hit = np.sum(played_predicted & played_expected)

    visual_missed = np.zeros_like(expected)
    mask = played_expected & ~played_predicted
    visual_missed[mask] = expected[mask]

    #plot_output_probs("Missed diff", 0.03225806451612903, visual_missed)
    #plot_output_probs("Predicted", 0.03225806451612903, predicted)
    #plot_output_probs("Expected", 0.03225806451612903, expected)
    #plt.show(block = True)

    hit_rate = 1.0
    if notes_hit + phantom_notes_diff + missed_notes_diff > 0:
        hit_rate = (notes_hit / (notes_hit + phantom_notes_diff + missed_notes_diff))

    visualization = None
    if generate_visualization:
        cmap = 'viridis'
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        X = jnp.linspace(0.0, predicted.shape[0], predicted.shape[0])
        Y = jnp.arange(MIDI_EVENT_VOCCAB_SIZE)
        c = ax1.pcolor(X, Y, jnp.transpose(output_probs), cmap=cmap, norm=norm)
        ax1.set(
            ylabel="Inferred events",
        )

        ax2.pcolor(X, Y, jnp.transpose(expected), cmap=cmap, norm=norm)
        ax2.set(
            xlabel="Time [frame]",
            ylabel="Expected events",
        )
        visualization = fig

    return DetailedEventLoss(
        full_diff=full_diff,
        phantom_notes_diff=phantom_notes_diff,
        missed_notes_diff=missed_notes_diff,
        notes_hit=notes_hit,
        hit_rate=hit_rate,
        visualization=visualization,
    )

def plot_prob_dist(quantity: str, dist: Float[Array, "len"]):
    fig, ax1 = plt.subplots()

    X = jnp.arange(dist.shape[0])
    ax1.plot(X, dist)

    ax1.set(
        xlabel=quantity,
        ylabel="Probability",
        title=f"Probability distribution for {quantity}",
    )

def load_newest_checkpoint(checkpoint_path: Path, ensemble_size: int = 1, ensemble_select: int | None = 0, model_replication=True):
    num_devices = len(jax.devices())

    # The randomness does not matter as we will load a checkpoint model anyways
    key = jax.random.PRNGKey(1234)

    # Setup model ensemble    
    @eqx.filter_vmap(out_axes=(eqx.if_array(0), eqx.if_array(0)))
    def make_ensemble(key):
        return eqx.nn.make_with_state(OutputSequenceGenerator)(model_config, key)
    ensemble_keys = jax.random.split(key, ensemble_size)
    audio_to_midi, state = make_ensemble(ensemble_keys) 
    # print(audio_to_midi)

    checkpoint_manager = ocp.CheckpointManager(checkpoint_path, item_names=('params', 'state'))
    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is None:
        raise "There is no checkpoint to load! Inference will be useless"
    else:
        current_metadata = get_model_metadata()
        if current_metadata != checkpoint_manager.metadata():
            print(f"WARNING: The loaded model has metadata {checkpoint_manager.metadata()}")
            print(f"Current configuration is {current_metadata}")

    print(f"Restoring saved model at step {step_to_restore}")
    model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
    restored_map = checkpoint_manager.restore(
        step_to_restore,
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(model_params),
            state=ocp.args.StandardRestore(state),
        ),
    )
    print(f"Read model parameters...")
    model_params = restored_map["params"]
    state = restored_map["state"]

    # Select the ensemble member to use
    if ensemble_select is not None:
        def ensemble_selector(path, x):
            if not eqx.is_array(x):
                # print(f"Skipping at {path} as it is not an array, value: {x}")
                return x

            # print(f"Selecting at {path} value {x}")
            return x[ensemble_select, ...]

        model_params = jax.tree.map_with_path(ensemble_selector, model_params)
        state = jax.tree.map_with_path(ensemble_selector, state)

    audio_to_midi = eqx.combine(model_params, eqx.filter(OutputSequenceGenerator(model_config, key), lambda x: not eqx.is_array(x)))

    if model_replication:
        # Replicate the model on all JAX devices
        device_mesh = mesh_utils.create_device_mesh((num_devices,))
        mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
        replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())

        model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
        model_params = jax.device_put(model_params, replicate_everywhere)
        audio_to_midi = eqx.combine(model_params, static_model)

    audio_to_midi = change_fp_precision(audio_to_midi, dtype=jnp.float32)
    audio_to_midi = eqx.nn.inference_mode(audio_to_midi)
    return audio_to_midi, state


def load_test_set(testset_dir: Path, num_model_output_frames: int):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)

    batches = []
    for sample_name in sample_names:
        midi_events, audio, _sample_names = AudioToMidiDatasetLoader.load_samples(testset_dir, num_model_output_frames, [sample_name], AudioToMidiDatasetLoader.SAMPLE_RATE, MODEL_AUDIO_LENGTH, skip_cache=True)
        batches.append((sample_name, audio, midi_events))
    return batches

def compute_testset_loss(
    model,
    state,
    rope_freqs: RopeFreqs,
    testset_dir: Path,
    num_model_output_frames: int,
):
    batches = load_test_set(testset_dir, num_model_output_frames)
    print("Loaded test set")

    @eqx.filter_jit(donate="all-except-first")
    def testset_loss_function(logits, expected_output):
        loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(logits, expected_output)
        return jnp.sum(loss)

    @eqx.filter_jit
    def run_inference_single_model(inference_model, state, audio, midi_events):
        # Compute the full batched model, even though we only compute one audio sample
        pretend_batch_audio = jnp.zeros((1, *audio.shape))
        pretend_batch_audio = pretend_batch_audio.at[0, ...].set(audio)

        pretend_batch_midi_events = jnp.zeros((1, *midi_events.shape))
        pretend_batch_midi_events = pretend_batch_midi_events.at[0, ...].set(midi_events)

        (logits, probs), _new_state = jax.vmap(inference_model, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch")(pretend_batch_audio, state, rope_freqs)
        test_losses = jax.vmap(testset_loss_function)(logits, pretend_batch_midi_events)
        return logits, probs, test_losses

    generate_visualizations = len(batches) < 30

    inference_model = eqx.nn.inference_mode(model)
    loss_map = {}
    for sample_name, audios, midi_events in batches:
        logits_all = []
        probs_all = []
        test_losses_all = []
        for audio, midi_event in zip(audios, midi_events):
            logits, probs, test_losses = run_inference_single_model(inference_model, state, audio, midi_event)
            logits_all.append(logits)
            probs_all.append(probs)
            test_losses_all.append(test_losses)

        stitched_probs = np.concatenate(probs, axis=0)
        stitched_events = np.concatenate(midi_events, axis=0)

        # TODO(knielsen): Consider exposing more information
        detailed_loss = detailed_event_loss(stitched_probs, stitched_events, generate_visualization=generate_visualizations)
        loss_map[sample_name] = {
            "loss": np.mean(test_losses_all),
        }

    print("Finished evaluating test loss")
    return loss_map


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process audio file to generate MIDI data.")

    # Add an argument for the input file
    parser.add_argument("input_file", help="Path to the input audio file.")
    parser.add_argument("--midi", help="Path to the output MIDI file.", default=None)
    parser.add_argument("--validation", action="store_true", help="Instruct the CLI to expect a directory and calculate the validation loss", default=None)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input file path from the parsed arguments
    input_file = args.input_file

    model, state = load_newest_checkpoint("/Volumes/git/ml/audio-to-midi/audio_to_midi_checkpoints")  # Assuming this function exists

    if args.validation:
        # Calculate and report the validation loss for the directory
        print(f"Calculating validation loss for {input_file}")
        rope_freqs = precompute_frequencies(model_config["attention_size"], 300)
        window_size = int(MODEL_AUDIO_LENGTH * AudioToMidiDatasetLoader.SAMPLE_RATE)
        num_model_output_frames = model.predict(state, jnp.zeros((2, window_size), jnp.float32), rope_freqs)[1].shape[0]
        # print(f"Model output frames: {num_model_output_frames}")
        loss_map = compute_testset_loss(
            model,
            state,
            rope_freqs,
            input_file,
            num_model_output_frames,
        )
        losses = np.stack([v["loss"] for v in loss_map.values()])
        print("Average loss: ", np.mean(losses))
        return

    # Infer a single file
    overlap = 0.25
    windowed_samples, window_duration = AudioToMidiDatasetLoader.load_and_slice_full_audio(input_file, overlap=overlap)  # Assuming this function exists
    _probs, stitched_probs, duration_per_frame = predict_and_stitch(model, state, windowed_samples, window_duration, overlap=overlap) # Assuming this function exists

    print(f"Frame count: {stitched_probs.shape[0]}")
    events = modelutil.extract_events(np.array(stitched_probs))
    frame_events = modelutil.to_frame_events([events], stitched_probs.shape[0])[0]
    print(f"Events: {events}")
    print(f"Frame events: {frame_events}")

    if args.midi:
        write_midi_file(events, duration_per_frame, args.midi)

    plot_output_probs("Inferred probs", duration_per_frame, stitched_probs)
    plt.show(block = True)

    return

if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.threefry_partitionable(True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
