from functools import partial, lru_cache
from pathlib import Path
from typing import Optional, Callable
import os
import csv
import time

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float
from functools import reduce
from more_itertools import chunked

from audio_to_midi_dataset import AudioToMidiDatasetLoader, visualize_sample
from model import OutputSequenceGenerator, model_config, get_model_metadata

@eqx.filter_jit
def compute_loss_from_output(logits, expected_output):
    # TODO: Get rid of this hack!
    #       This is due to the convolution shrinking the output logits.
    #       This should be handled in a better way...
    expected_output = expected_output[:logits.shape[0], ...]
    loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(logits, expected_output)
    return jnp.sum(loss)

@eqx.filter_jit
@eqx.filter_value_and_grad
def compute_loss(model, audio_frames, expected_outputs, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    logits, probs = jax.vmap(
        model, in_axes=(0, 0, None)
    )(audio_frames, batched_keys, True)

    loss = jax.vmap(compute_loss_from_output)(logits, expected_outputs)
    return jnp.mean(loss)

@eqx.filter_jit
def compute_training_step(
    flat_model, audio_frames, expected_outputs, flat_opt_state, key, tx,
    treedef_model, treedef_opt_state
):
    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

    # print(f"Audio frames shape: {audio_frames.shape}")
    # print(f"Expected outputs shape: {expected_outputs.shape}")

    key, new_key = jax.random.split(key)
    loss, grads = compute_loss(
        model,
        audio_frames=audio_frames,
        expected_outputs=expected_outputs,
        key=key,
    )

    updates, update_opt_state = tx.update(grads, opt_state, model)
    update_model = eqx.apply_updates(model, updates)

    flat_update_model = jax.tree_util.tree_leaves(update_model)
    flat_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)

    return loss, flat_update_model, flat_update_opt_state, new_key


@lru_cache(maxsize=1)
def load_test_set(testset_dir: Path, sharding, batch_size: int):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)
    chunks = chunked(sample_names, batch_size)

    batches = []
    for chunk in chunks:
        midi_events, _, frames, duration_per_frame, frame_width = AudioToMidiDatasetLoader.load_samples(testset_dir, chunk, minimum_midi_event_size=128, sharding=sharding)
        batches.append((chunk, frames, midi_events))
    return batches

@eqx.filter_jit
def compute_test_loss(
    model,
    key: jax.random.PRNGKey,
    audio_frames: Float[Array, "num_frames num_frame_data"],
    midi_events: Float[Array, "num_frames midi_voccab_size"]
):
    logits, probs = model(audio_frames, key)
    return compute_loss_from_output(logits, midi_events)

def compute_testset_loss_individual(model, testset_dir: Path, key: jax.random.PRNGKey, sharding, batch_size=32):
    batches = load_test_set(testset_dir, sharding, batch_size=batch_size)
    print("Loaded test set")

    loss_map = {}
    for sample_names, frames, midi_events in batches:
        test_loss_keys = jax.random.split(key, num=frames.shape[0])
        test_losses = jax.vmap(compute_test_loss, (None, 0, 0, 0))(model, test_loss_keys, frames, midi_events)
        for sample_name, loss in zip(sample_names, test_losses):
            loss_map[sample_name] = { "loss": loss }

    print("Finished evaluating test loss")
    return loss_map

def compute_testset_loss(model, testset_dir: Path, key: jax.random.PRNGKey, sharding, batch_size=32):
    per_sample_map = compute_testset_loss_individual(model, testset_dir, key, sharding, batch_size)

    test_loss = jnp.zeros((1,))
    count = jnp.array(0, dtype=jnp.int32)
    for losses in per_sample_map.values():
        test_loss += losses["loss"]
        count += 1

    return (test_loss / count)[0]


def train(
    model,
    tx,
    data_loader,
    state: optax.OptState,
    checkpoint_manager: ocp.CheckpointManager,
    trainloss_csv: Optional[any],
    testloss_csv: Optional[any],
    learning_rate_schedule: Callable,
    device_mesh: [],
    testset_dir: Optional[Path],
    num_steps: int = 10000,
    print_every: int = 1000,
    testset_loss_every: int = 1000,
    key: Optional[jax.random.PRNGKey] = None,
):
    start_time = time.time()

    start_step = (
        checkpoint_manager.latest_step() + 1
        if checkpoint_manager.latest_step() is not None
        else 0
    )

    batch_mesh = Mesh(device_mesh, ("batch",))
    batch_sharding = NamedSharding(
        batch_mesh,
        PartitionSpec(
            "batch",
        ),
    )

    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_state, treedef_state = jax.tree_util.tree_flatten(state)

    loss_sum = jnp.array([0.0], dtype=jnp.float32)

    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        (audio_frames, events) = jax.device_put(
            (batch["audio_frames"], batch["events"]),
            batch_sharding,
        )

        # Keep the old model state in memory until we are sure the loss is not nan
        recovery_flat_model = flat_model
        recovery_flat_state = flat_state

        loss, flat_model, flat_state, key = compute_training_step(
            flat_model,
            audio_frames,
            events,
            flat_state,
            key,
            tx,
            treedef_model,
            treedef_state,
        )
        step_end_time = time.time()

        if jnp.isnan(loss):
            print(f"Encountered NAN loss at step {step}. Trying to recover!")
            flat_model = recovery_flat_model
            flat_state = recovery_flat_state
            continue

        if checkpoint_manager.should_save(step):
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            filtered_model = eqx.filter(model, eqx.is_array)
            checkpoint_manager.save(step, args=ocp.args.StandardSave(filtered_model))

        loss_sum = loss_sum + loss

        if step % print_every == 0 and step != 0:
            learning_rate = learning_rate_schedule(step)

            averaged_loss = (loss_sum / print_every)[0]

            if trainloss_csv is not None:
                trainloss_csv.writerow([step, averaged_loss, step_end_time - start_time, step * audio_frames.shape[0], learning_rate])
            print(f"Step {step}/{num_steps}, Loss: {averaged_loss}, LR = {learning_rate}")

            loss_sum = jnp.array([0.0], dtype=jnp.float32)

        if step % testset_loss_every == 0 and testset_dir is not None:
            print("Evaluating test loss...")
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            eval_key, key = jax.random.split(key, num=2)
            testset_loss = compute_testset_loss(model, testset_dir, eval_key, batch_sharding)
            if testloss_csv is not None:
                testloss_csv.writerow([step, testset_loss, step_end_time - start_time, step * audio_frames.shape[0]])
            print(f"Test loss: {testset_loss}")

    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    state = jax.tree_util.tree_unflatten(treedef_state, flat_state)
    return model, state

def create_learning_rate_schedule(base_learning_rate: float, warmup_steps: int, cosine_decay_steps: int):
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=base_learning_rate,
        transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_decay_steps
    )
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )

def main():
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/true_harmonic")
    testset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set")

    num_devices = len(jax.devices())

    batch_size = 4 * num_devices
    num_steps = 1000000
    learning_rate_schedule = create_learning_rate_schedule(2.5 * 1e-4, 1000, num_steps)

    checkpoint_every = 200
    checkpoints_to_keep = 3
    dataset_prefetch_count = 20

    key = jax.random.PRNGKey(1234)
    model_init_key, training_key, dataset_loader_key = jax.random.split(key, num=3)

    print(f"Running on {num_devices} devices with an effective batch size of {batch_size}")

    audio_to_midi = OutputSequenceGenerator(model_config, model_init_key)

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())

    model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
    model_params = jax.device_put(model_params, replicate_everywhere)
    audio_to_midi = eqx.combine(model_params, static_model)

    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpoints_to_keep, save_interval_steps=checkpoint_every
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path,
        options=checkpoint_options,
        metadata=get_model_metadata()
    )

    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is not None:
        print(f"Restoring saved model at step {step_to_restore}")
        model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
        model_params = checkpoint_manager.restore(
            step_to_restore,
            args=ocp.args.StandardRestore(model_params),
        )
        audio_to_midi = eqx.combine(model_params, static_model)

    tx = optax.adamw(learning_rate=learning_rate_schedule)
    tx = optax.chain(optax.clip_by_global_norm(5.0), tx)
    # The filtering is necessary to have the opt-state flattening working
    state = tx.init(eqx.filter(audio_to_midi, eqx.is_inexact_array))

    print("Setting up dataset loader...")
    dataset_loader = AudioToMidiDatasetLoader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        prefetch_count=dataset_prefetch_count,
        key=dataset_loader_key,
    )
    dataset_loader_iter = iter(dataset_loader)

    print("Starting training...")
    with open('train_loss.csv', mode='w', buffering=1) as trainloss_file:
        with open('test_loss.csv', mode='w', buffering=1) as testloss_file:
            trainloss_csv = csv.writer(trainloss_file)
            testloss_csv = csv.writer(testloss_file)

            audio_to_midi, state = train(
                audio_to_midi,
                tx,
                dataset_loader_iter,
                state,
                checkpoint_manager,
                trainloss_csv=trainloss_csv,
                testloss_csv=testloss_csv,
                learning_rate_schedule=learning_rate_schedule,
                device_mesh=device_mesh,
                testset_dir=testset_dir,
                num_steps=num_steps,
                print_every=5,
                key=training_key,
                testset_loss_every=checkpoint_every,
            )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace"):
    main()
