from pathlib import Path

import os
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float

from audio_to_midi_dataset import (
    BLANK_MIDI_EVENT,
    BLANK_VELOCITY,
    BLANK_DURATION,
    SEQUENCE_END,
    SEQUENCE_START,
    NUM_VELOCITY_CATEGORIES,
    AudioToMidiDatasetLoader,
    plot_frequency_domain_audio,
)
from model import OutputSequenceGenerator, model_config, get_model_metadata


@eqx.filter_jit
def forward(model, audio_frames, outputs_so_far, key):
    inference_keys = jax.random.split(key, num=audio_frames.shape[0])
    return jax.vmap(
        model, (0, 0, 0)
    )(audio_frames, outputs_so_far, inference_keys)


def _update_raw_outputs(
    current,
    midi_logits,
    midi_probs,
    attack_time_logits,
    attack_time_probs,
    duration_logits,
    duration_probs,
    velocity_logits,
    velocity_probs,
):
    midi_logits = midi_logits[:, None, :]
    midi_probs = midi_probs[:, None, :]
    attack_time_logits = attack_time_logits[:, None, :]
    attack_time_probs = attack_time_probs[:, None, :]
    duration_logits = duration_logits[:, None, :]
    duration_probs = duration_probs[:, None, :]
    velocity_logits = velocity_logits[:, None, :]
    velocity_probs = velocity_probs[:, None, :]

    if current is None:
        return {
            "midi_logits": midi_logits,
            "midi_probs": midi_probs,
            "attack_time_logits": attack_time_logits,
            "attack_time_probs": attack_time_probs,
            "duration_logits": duration_logits,
            "duration_probs": duration_probs,
            "velocity_logits": velocity_logits,
            "velocity_probs": velocity_probs,
        }

    return {
        "midi_logits": jnp.concatenate([current["midi_logits"], midi_logits], axis=1),
        "midi_probs": jnp.concatenate([current["midi_probs"], midi_probs], axis=1),
        "attack_time_logits": jnp.concatenate(
            [current["attack_time_logits"], attack_time_logits], axis=1
        ),
        "attack_time_probs": jnp.concatenate(
            [current["attack_time_probs"], attack_time_probs], axis=1
        ),
        "duration_logits": jnp.concatenate(
            [current["duration_logits"], duration_logits], axis=1
        ),
        "duration_probs": jnp.concatenate(
            [current["duration_probs"], duration_probs], axis=1
        ),
        "velocity_logits": jnp.concatenate(
            [current["velocity_logits"], velocity_logits], axis=1
        ),
        "velocity_probs": jnp.concatenate(
            [current["velocity_probs"], velocity_probs], axis=1
        ),
    }


def batch_infer(
    model,
    key: jax.random.PRNGKey,
    frames: Float[Array, "batch_size frame_count"],
    infer_limit: int = 50,
):
    batch_size = frames.shape[0]
    seen_events = jnp.tile(
        jnp.array([0, SEQUENCE_START, BLANK_DURATION, BLANK_VELOCITY], dtype=jnp.int16),
        (batch_size, 1, 1),
    )

    i = 0
    # This will be an all False array in the beginning. The mask is updated every step in the inference loop
    end_of_sequence_mask = (seen_events[:, -1, 1] == SEQUENCE_END) | (
        seen_events[:, -1, 1] == BLANK_MIDI_EVENT
    )

    raw_outputs = None

    # TODO: Consider jitting the inside of this loop
    padding_increment = 20
    while (not jnp.all(end_of_sequence_mask)) and i < infer_limit:
        inference_key, key = jax.random.split(key, num=2)

        # Inference result should not change appending padding events!
        # Use a padded version to avoid jax recompilations
        padding_amount = (padding_increment - seen_events.shape[1]) % padding_increment
        padding = jnp.tile(
            jnp.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], dtype=jnp.int16), (batch_size, padding_amount, 1)
        )
        padded_seen_events = jnp.concatenate([seen_events, padding], axis=1)
        jax.debug.print(
            "Padded seen events {padded_seen_events}",
            padded_seen_events=padded_seen_events,
        )

        (
            midi_logits,
            midi_probs,
            attack_time_logits,
            attack_time_probs,
            duration_logits,
            duration_probs,
            velocity_logits,
            velocity_probs,
        ) = forward(
            model,
            frames,
            padded_seen_events,
            inference_key,
        )

        notes = jnp.select(
            [end_of_sequence_mask],
            [jnp.tile(jnp.array(BLANK_MIDI_EVENT, dtype=jnp.int16), (batch_size,))],
            jnp.argmax(midi_probs, axis=1),
        )

        # Make sure the position is always monotonically increasing
        # TODO: Consider to throw a weighted dice with position_probs probabilities here?
        attack_times = jnp.maximum(
            seen_events[:, -1, 0], jnp.argmax(attack_time_probs, axis=1)
        )
        attack_times = jnp.select(
            [end_of_sequence_mask],
            [jnp.zeros((batch_size,), jnp.int16)],
            attack_times,
        )

        durations = jnp.select(
             [end_of_sequence_mask],
             [jnp.zeros((batch_size,), jnp.int16)],
             jnp.argmax(duration_probs, axis=1),
         )

        velocities = jnp.select(
             [end_of_sequence_mask],
             [jnp.zeros((batch_size,), jnp.int16)],
             jnp.argmax(velocity_probs, axis=1),
         )

        # Combine the predicted positions with their corresponding midi events
        predicted_events = jnp.transpose(
            jnp.vstack([attack_times, notes, durations, velocities])
        )

        # Update seen events with the new predictions
        seen_events = jnp.concatenate(
            [seen_events, jnp.reshape(predicted_events, (batch_size, 1, 4))], axis=1
        )
        # print(f"Seen events at step {i}", seen_events)

        # Update logits and probs
        raw_outputs = _update_raw_outputs(
            raw_outputs,
            midi_logits,
            midi_probs,
            attack_time_logits,
            attack_time_probs,
            raw_durations,
            raw_velocities,
        )

        end_of_sequence_mask = (seen_events[:, -1, 1] == SEQUENCE_END) | (
            seen_events[:, -1, 1] == BLANK_MIDI_EVENT
        )
        i += 1

    return seen_events, raw_outputs


def plot_prob_dist(quantity: str, dist: Float[Array, "len"]):
    fig, ax1 = plt.subplots()

    X = jnp.arange(dist.shape[0])
    ax1.plot(X, dist)

    ax1.set(
        xlabel=quantity,
        ylabel="Probability",
        title=f"Probability distribution for {quantity}",
    )

def load_newest_checkpoint(checkpoint_path: Path):
    num_devices = len(jax.devices())

    # The randomness does not matter as we will load a checkpoint model anyways
    key = jax.random.PRNGKey(1234)
    audio_to_midi = OutputSequenceGenerator(model_config, key)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_path)

    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is None:
        raise "There is no checkpoint to load! Inference will be useless"
    else:
        current_metadata = get_model_metadata()
        if current_metadata != checkpoint_manager.metadata():
            print(f"WARNING: The loaded model has metadata {checkpoint_manager.metadata()}, but current configuration is {current_metadata}")

    print(f"Restoring saved model at step {step_to_restore}")
    model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
    model_params = checkpoint_manager.restore(
        step_to_restore,
        args=ocp.args.StandardRestore(model_params),
    )
    audio_to_midi = eqx.combine(model_params, static_model)

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())

    model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
    model_params = jax.device_put(model_params, replicate_everywhere)
    audio_to_midi = eqx.combine(model_params, static_model)

    return audio_to_midi


def main():
    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")

    key = jax.random.PRNGKey(1234)
    inference_key, dataset_loader_key, test_loss_key = jax.random.split(key, num=3)

    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    audio_to_midi = load_newest_checkpoint(checkpoint_path)

    print("Loading audio file...")
    # TODO: Handle files that are longer than 5 seconds
    # TODO: Support loading a file from a CLI argument
    sample_names = ["piano_BechsteinFelt_48", "piano_BechsteinFelt_70"]
    (
        all_frames,
        sample_rate,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames_from_sample_name(dataset_dir, sample_names)
    # print(f"Frames shape: {all_frames.shape}")

    plot_frequency_domain_audio(duration_per_frame, all_frames[0])
    plot_frequency_domain_audio(duration_per_frame, all_frames[1])

    print("Infering midi events...")
    inferred_events, raw_outputs = batch_infer(audio_to_midi, inference_key, all_frames)
    print(f"Inferred events: {inferred_events}")

    # # Evaluate losses

    expected_midi_events = (
        AudioToMidiDatasetLoader.load_midi_events_frame_time_positions(
            dataset_dir, sample_names, duration_per_frame
        )
    )

    from train import compute_test_loss, compute_testset_loss
    loss = compute_test_loss(audio_to_midi, test_loss_key, all_frames[0], expected_midi_events[0])
    print(f"Loss of example: {loss}")

    testset_loss = compute_testset_loss(audio_to_midi, dataset_dir, test_loss_key)
    print(f"Testset loss: {testset_loss}")

    # # For now just compute the loss of the first example
    i = 0
    end_of_sequence_mask = (inferred_events[:, :, 1] == SEQUENCE_END) | (
        inferred_events[:, :, 1] == BLANK_MIDI_EVENT
    )
    from train import compute_loss_from_output

    while not jnp.all(end_of_sequence_mask[:, i]):
        loss, individual_losses = compute_loss_from_output(
            raw_outputs["midi_logits"][:, i, :],
            raw_outputs["position_probs"][:, i, :],
            raw_outputs["velocity_probs"][:, i, :],
            expected_midi_events[:, i + 1],
            all_frames[0].shape[0],
        )
        loss = ~end_of_sequence_mask[:, i] * loss
        print(f"Loss at step {i}: {loss}")

        plot_prob_dist("position", raw_outputs["position_probs"][0, i, :])
        plot_prob_dist("velocity", raw_outputs["velocity_probs"][0, i, :])
        plt.show()

        i += 1

    plt.show()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
