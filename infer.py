from pathlib import Path

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from audio_to_midi_dataset import (
    SEQUENCE_END,
    SEQUENCE_START,
    AudioToMidiDatasetLoader,
    plot_frequency_domain_audio,
)
from model import OutputSequenceGenerator
from train import model_config


@eqx.filter_jit
def infer(model, audio_frames, outputs_so_far, key):
    midi_logits, midi_probs, position_logits, position_probs = model(
        audio_frames, outputs_so_far, key
    )
    return midi_logits, midi_probs, position_logits, position_probs


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
        frames,
        sample_rate,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames(
        dataset_dir / "piano_BechsteinFelt_48.aac"
    )

    plot_frequency_domain_audio(duration_per_frame, frames)

    print("Infering midi events...")
    last_event = (0, SEQUENCE_START)
    seen_events = jnp.array([last_event])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])
    # seen_events = jnp.vstack([seen_events, jnp.array([-1, 0])])

    infer_limit = 50
    i = 0
    while last_event[1] != SEQUENCE_END and i < infer_limit:
        print(f"Seen events: {seen_events}")
        _, midi_probs, _, position_probs = infer(
            audio_to_midi,
            frames,
            seen_events,
            inference_key,
        )

        midi_event = jnp.argmax(midi_probs)
        position = jnp.argmax(position_probs)
        last_event = (position, midi_event)

        print(f"{position}\t{midi_event}")

        seen_events = jnp.vstack(
            (seen_events, jnp.array([last_event])), dtype=jnp.int16
        )
        i += 1

    if i == infer_limit:
        print("HIT INFERENCE LIMIT!!!")

    plt.show()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()

    # https://github.com/microsoft/vscode/issues/174295 - ¯\_(ツ)_/¯
