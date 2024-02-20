from pathlib import Path

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
    SEQUENCE_END,
    SEQUENCE_START,
    AudioToMidiDatasetLoader,
    plot_frequency_domain_audio
)
from model import OutputSequenceGenerator, model_config


@eqx.filter_jit
def forward(model, audio_frames, outputs_so_far, key):
    inference_keys = jax.random.split(key, num=audio_frames.shape[0])
    midi_logits, midi_probs, position_logits, position_probs, velocity_logits, velocity_probs = jax.vmap(
        model, (0, 0, 0)
    )(audio_frames, outputs_so_far, inference_keys)
    return midi_logits, midi_probs, position_logits, position_probs, velocity_logits, velocity_probs


def _update_raw_outputs(
    current,
    midi_logits,
    midi_probs,
    position_logits,
    position_probs,
    velocity_logits,
    velocity_probs,
):
    midi_logits = midi_logits[:, None, :]
    midi_probs = midi_probs[:, None, :]
    position_logits = position_logits[:, None, :]
    position_probs = position_probs[:, None, :]
    velocity_logits = velocity_logits[:, None, :]
    velocity_probs = velocity_probs[:, None, :]

    if current is None:
        return {
            "midi_logits": midi_logits,
            "midi_probs": midi_probs,
            "position_logits": position_logits,
            "position_probs": position_probs,
            "velocity_logits": velocity_logits,
            "velocity_probs": velocity_probs,
        }

    return {
        "midi_logits": jnp.concatenate([current["midi_logits"], midi_logits], axis=1),
        "midi_probs": jnp.concatenate([current["midi_probs"], midi_probs], axis=1),
        "position_logits": jnp.concatenate(
            [current["position_logits"], position_logits], axis=1
        ),
        "position_probs": jnp.concatenate(
            [current["position_probs"], position_probs], axis=1
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
    infer_limit: int = 20,
):
    batch_size = frames.shape[0]
    seen_events = jnp.tile(
        jnp.array([0, SEQUENCE_START, BLANK_VELOCITY], dtype=jnp.int16),
        (batch_size, 1, 1),
    )

    i = 0
    # This will be an all False array in the beginning. The mask is updated every step in the inference loop
    end_of_sequence_mask = (seen_events[:, -1, 1] == SEQUENCE_END) | (
        seen_events[:, -1, 1] == BLANK_MIDI_EVENT
    )

    raw_outputs = None

    # TODO: Consider jitting the inside of this loop
    while (not jnp.all(end_of_sequence_mask)) and i < infer_limit:
        inference_key, key = jax.random.split(key, num=2)

        # Inference result should not change appending padding events!
        # padding = jnp.tile(
        #     jnp.array([0, BLANK_MIDI_EVENT], dtype=jnp.int16), (batch_size, 5, 1)
        # )
        # padded_seen_events = jnp.concatenate([seen_events, padding], axis=1)
        # jax.debug.print(
        #     "Padded seen events {padded_seen_events}",
        #     padded_seen_events=padded_seen_events,
        # )
        (
            midi_logits,
            midi_probs,
            position_logits,
            position_probs,
            velocity_logits,
            velocity_probs,
        ) = forward(
            model,
            frames,
            seen_events,
            inference_key,
        )

        midi_events = jnp.select(
            [end_of_sequence_mask],
            [jnp.tile(jnp.array(BLANK_MIDI_EVENT, dtype=jnp.int16), (batch_size,))],
            jnp.argmax(midi_probs, axis=1),
        )

        # Make sure the position is always monotonically increasing
        # TODO: Consider to throw a weighted dice with position_probs probabilities here?
        positions = jnp.maximum(
            seen_events[:, -1, 0], jnp.argmax(position_probs, axis=1)
        )
        positions = jnp.select(
            [end_of_sequence_mask],
            [jnp.zeros((batch_size,), jnp.int16)],
            positions,
        )

        velocities = jnp.select(
            [end_of_sequence_mask],
            [jnp.zeros((batch_size,), jnp.int16)],
            jnp.argmax(velocity_probs, axis=1),
        )

        # Combine the predicted positions with their corresponding midi events
        predicted_events = jnp.transpose(
            jnp.vstack([positions, midi_events, velocities])
        )

        # Update seen events with the new predictions
        seen_events = jnp.concatenate(
            [seen_events, jnp.reshape(predicted_events, (batch_size, 1, 3))], axis=1
        )
        # print(f"Seen events at step {i}", seen_events)

        # Update logits and probs
        raw_outputs = _update_raw_outputs(
            raw_outputs,
            midi_logits,
            midi_probs,
            position_logits,
            position_probs,
            velocity_logits,
            velocity_probs,
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


def main():
    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")

    num_devices = len(jax.devices())

    key = jax.random.PRNGKey(1234)
    model_init_key, inference_key, dataset_loader_key = jax.random.split(key, num=3)

    # TODO: Enable dropout for training
    audio_to_midi = OutputSequenceGenerator(model_config, model_init_key)

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())
    audio_to_midi = jax.device_put(audio_to_midi, replicate_everywhere)

    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    checkpoint_manager = ocp.CheckpointManager(checkpoint_path)

    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is not None:
        print(f"Restoring saved model at step {step_to_restore}")
        audio_to_midi = checkpoint_manager.restore(
            step_to_restore,
            args=ocp.args.StandardRestore(audio_to_midi),
        )

    print("Loading audio file...")
    # TODO: Handle files that are longer than 5 seconds
    # TODO: Support loading a file from a CLI argument
    sample_names = ["piano_BechsteinFelt_48", "piano_BechsteinFelt_70"]
    (
        all_frames,
        sample_rate,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames(dataset_dir, sample_names)
    # print(f"Frames shape: {all_frames.shape}")

    plot_frequency_domain_audio(duration_per_frame, all_frames[0])
    plot_frequency_domain_audio(duration_per_frame, all_frames[1])

    print("Infering midi events...")
    inferred_events, raw_outputs = batch_infer(audio_to_midi, inference_key, all_frames)
    print(f"Inferred events: {inferred_events}")

    # Evaluate losses

    expected_midi_events = (
        AudioToMidiDatasetLoader.load_midi_events_frame_time_positions(
            dataset_dir, sample_names, duration_per_frame
        )
    )

    # # For now just compute the loss of the first example
    i = 0
    end_of_sequence_mask = (inferred_events[:, :, 1] == SEQUENCE_END) | (
        inferred_events[:, :, 1] == BLANK_MIDI_EVENT
    )
    from train import compute_loss_from_output

    while not jnp.all(end_of_sequence_mask[:, i]):
        loss = compute_loss_from_output(
            raw_outputs["midi_logits"][:, i, :],
            raw_outputs["position_probs"][:, i, :],
            raw_outputs["velocity_probs"][:, i, :],
            expected_midi_events[:, i + 1],
        )
        loss = ~end_of_sequence_mask[:, i] * loss
        print(f"Loss at step {i}: {loss}")

        plot_prob_dist("position", raw_outputs["position_probs"][0, i, :])
        plot_prob_dist("velocity", raw_outputs["velocity_probs"][0, i, :])
        plt.show()

        i += 1

    plt.show()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
