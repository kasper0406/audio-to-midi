from pathlib import Path

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from audio_to_midi_dataset import AudioToMidiDatasetLoader, visualize_sample
from model import OutputSequenceGenerator
from train import model_config


@eqx.filter_jit
def infer(model, audio_frames, outputs_so_far, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    midi_logits, midi_probs, position_logits, position_probs = jax.vmap(
        model, in_axes=(0, 0, 0)
    )(audio_frames, outputs_so_far, batched_keys)

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

    print("Setting up dataset loader...")
    dataset_loader = AudioToMidiDatasetLoader(
        dataset_dir=dataset_dir,
        batch_size=1,
        prefetch_count=1,
        num_workers=1,
        key=dataset_loader_key,
    )
    dataset_loader_iter = iter(dataset_loader)

    sample_batch = next(dataset_loader_iter)
    visualize_sample(
        sample_batch["audio_frames"][0],
        sample_batch["seen_events"][0],
        sample_batch["next_event"][0],
        sample_batch["duration_per_frame_in_secs"],
    )

    print("Actual seen events", sample_batch["seen_events"].shape)
    # seen_events = jnp.zeros(shape=(1, 1, 2))
    # print("Faked seen events", seen_events.shape)

    midi_logits, midi_probs, _, _ = infer(
        audio_to_midi,
        sample_batch["audio_frames"],
        sample_batch["seen_events"],
        inference_key,
    )

    print("Midi logits", midi_logits)
    print("Midi probs", midi_probs)


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
