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
    SEQUENCE_END,
    SEQUENCE_START,
    AudioToMidiDatasetLoader,
    plot_frequency_domain_audio,
)
from model import OutputSequenceGenerator
from train import model_config


@eqx.filter_jit
def forward(model, audio_frames, outputs_so_far, key):
    inference_keys = jax.random.split(key, num=audio_frames.shape[0])
    midi_logits, midi_probs, position_logits, position_probs = jax.vmap(
        model, (0, 0, 0)
    )(audio_frames, outputs_so_far, inference_keys)
    return midi_logits, midi_probs, position_logits, position_probs


def batch_infer(
    model,
    key: jax.random.PRNGKey,
    frames: Float[Array, "batch_size frame_count"],
    infer_limit: int = 20,
):
    batch_size = frames.shape[0]
    seen_events = jnp.tile(
        jnp.array([0, SEQUENCE_START], dtype=jnp.int16), (batch_size, 1, 1)
    )

    i = 0
    # This will be an all False array in the beginning. The mask is updated every step in the inference loop
    end_of_sequence_mask = (seen_events[:, -1, 1] == SEQUENCE_END) | (
        seen_events[:, -1, 1] == BLANK_MIDI_EVENT
    )

    while (not jnp.all(end_of_sequence_mask)) and i < infer_limit:
        inference_key, key = jax.random.split(key, num=2)
        _, midi_probs, _, position_probs = forward(
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
        positions_no_pad = jnp.maximum(
            seen_events[:, -1, 0], jnp.argmax(position_probs, axis=1)
        )
        positions = jnp.select(
            [end_of_sequence_mask],
            [jnp.zeros((batch_size,), jnp.int16)],
            positions_no_pad,
        )

        # Combine the predicted positions with their corresponding midi events
        predicted_events = jnp.transpose(jnp.vstack([positions, midi_events]))

        # Update seen events with the new predictions
        seen_events = jnp.concatenate(
            [seen_events, jnp.reshape(predicted_events, (batch_size, 1, 2))], axis=1
        )
        # print(f"Seen events at step {i}", seen_events)

        end_of_sequence_mask = (seen_events[:, -1, 1] == SEQUENCE_END) | (
            seen_events[:, -1, 1] == BLANK_MIDI_EVENT
        )
        i += 1

    return seen_events


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
    (
        frames_1,
        sample_rate,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames(
        dataset_dir / "piano_BechsteinFelt_48.aac"
    )

    (
        frames_2,
        sample_rate,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames(
        dataset_dir / "piano_BechsteinFelt_70.aac"
    )

    plot_frequency_domain_audio(duration_per_frame, frames_1)
    plot_frequency_domain_audio(duration_per_frame, frames_2)

    all_frames = jnp.stack([frames_1, frames_2])

    print("Infering midi events...")

    # Inference result should not change appending padding events!
    # seen_events = jnp.vstack([seen_events, jnp.array([0, -1])])
    # seen_events = jnp.vstack([seen_events, jnp.array([0, -1])])
    # seen_events = jnp.vstack([seen_events, jnp.array([0, -1])])

    inferred_events = batch_infer(audio_to_midi, inference_key, all_frames)
    print(f"Inferred events: {inferred_events}")

    plt.show()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
