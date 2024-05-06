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
import numpy as np

from model import OutputSequenceGenerator, model_config, get_model_metadata

def stitch_output_probs(all_probs, duration_per_frame: float, overlap: float):
    # Append a frame at the beginning with the start of the first frame to make the stitching
    # below work out as intended
    overlapping_frames = int(overlap / duration_per_frame)
    replicated_frame = np.zeros((all_probs.shape[1], all_probs.shape[2]), dtype=np.float32)
    replicated_frame[-overlapping_frames:] += all_probs[0, :overlapping_frames, ...]
    all_probs = np.concatenate([ replicated_frame[np.newaxis, ...], all_probs ])

    output = np.zeros((0, all_probs.shape[2]), dtype=np.float32)
    for i in range(1, all_probs.shape[0]):
        overlap = (all_probs[i - 1, -overlapping_frames:, ...] + all_probs[i, :overlapping_frames]) / 2
        non_overlap = all_probs[i, (overlapping_frames + 1):-(overlapping_frames + 1), ...]
        output = np.concatenate([output, overlap, non_overlap])
    # For the last probs there are no overlap, so we just add the region directly
    output = np.concatenate([output, all_probs[-1, -overlapping_frames:, ...]])

    return output

def forward(model, audio_frames, key, duration_per_frame: float, overlap=0.0):
    inference_keys = jax.random.split(key, num=audio_frames.shape[0])
    _logits, probs = jax.vmap(model)(audio_frames, inference_keys)

    # HACK: Get rid of this!
    #       Currently we train the model to not output the last three frames, so the overlap will be different
    hacked_overlap = overlap - (audio_frames.shape[2] - probs.shape[1]) * duration_per_frame
    print(f"Hacked overlap: {hacked_overlap}")
    return probs, stitch_output_probs(probs, duration_per_frame, hacked_overlap)


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
    return

if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
