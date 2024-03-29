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

from audio_to_midi_dataset import BLANK_MIDI_EVENT, BLANK_VELOCITY, AudioToMidiDatasetLoader, get_active_events
from model import OutputSequenceGenerator, model_config


# List of ideas:
#  - Double check logic for event context windows
#  - Try to disable noise in audio frames
#  - Try to changge audio frame offset to 0, to maybe give attention mechanism an easier time?
#  - Give the model the list of currently playing notes
#  - No extra midi keys for releases - play note at velocity 0
#  - Try different optimizer (LAMB) allowing larger batch sizes on GPU



@eqx.filter_jit
def continous_probability_loss(probs, expected, variance):
    """
    Some kind of loss function that will promote having approximately correct positions / velocities

    Assumption: The value of the integer `expected` should be in the range 0 <= expected <= probs.shape[0]!
    """
    epsilon = 0.0000001
    x = jnp.arange(probs.shape[0])
    expectation = -0.5 * jnp.square((x - expected) / variance)
    expectation = jnp.maximum(
        expectation, -10.0
    )  # Below values of -10 we will assume the exponential will give 0
    expectation = jnp.exp(expectation)
    expectation = expectation / jnp.sum(expectation)
    # jax.debug.print("Expectation: {expectation}", expectation=expectation)
    return optax.kl_divergence(jnp.log(probs + epsilon), expectation)

@eqx.filter_jit
def velocity_loss_fn(probs, expected, variance):
    epsilon = 0.0000001

    expectation_if_zero = jnp.zeros(probs.shape[0])
    expectation_if_zero = expectation_if_zero.at[0].set(1.0)

    x = jnp.arange(probs.shape[0])
    expectation_if_nonzero = -0.5 * jnp.square((x - expected) / variance)
    expectation_if_nonzero = jnp.maximum(
        expectation_if_nonzero, -10.0
    )  # Below values of -10 we will assume the exponential will give 0
    expectation_if_nonzero = jnp.exp(expectation_if_nonzero)
    # 0 probability of velocity 0 as that would be no event in the first place
    expectation_if_nonzero = expectation_if_nonzero.at[0].set(0.0)

    expectation = jnp.select([expected == 0], [expectation_if_zero], expectation_if_nonzero)
    expectation = expectation / jnp.sum(expectation)
    # jax.debug.print("Expectation: {expectation}", expectation=expectation)
    return optax.kl_divergence(jnp.log(probs + epsilon), expectation)

@eqx.filter_jit
def compute_loss_from_output(
    midi_logits, position_probs, velocity_probs, expected_next_output
):
    expected_next_midi = expected_next_output[:, 1]
    midi_event_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=midi_logits, labels=expected_next_midi
    )
    # TODO: Consider if this can be represented in some other way
    expected_next_position = expected_next_output[:, 0]
    position_loss = jax.vmap(partial(continous_probability_loss, variance=3.0), (0, 0))(
        position_probs, expected_next_position
    )

    expected_next_velocity = expected_next_output[:, 2]
    velocity_loss = jax.vmap(partial(velocity_loss_fn, variance=1.0), (0, 0))(
        velocity_probs, expected_next_velocity
    )

    # TODO: Fix the weight on the position loss so it is not hard-coded, but part of the config
    individual_losses = jnp.array([ midi_event_loss, position_loss, velocity_loss ])
    return midi_event_loss + 0.3 * position_loss + 0.2 * velocity_loss, individual_losses


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(model, audio_frames, outputs_so_far, active_events, expected_next_output, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    midi_logits, _, _, position_probs, _, velocity_probs = jax.vmap(
                model, in_axes=(0, 0, 0, 0)
    )(audio_frames, outputs_so_far, active_events, batched_keys)

    loss, individual_losses = compute_loss_from_output(
        midi_logits, position_probs, velocity_probs, expected_next_output
    )
    return jnp.mean(loss), jnp.mean(individual_losses, axis=1)


@eqx.filter_jit
def compute_training_step(
    flat_model, audio_frames, outputs_so_far, active_events, next_output, flat_opt_state, key, tx,
    treedef_model, treedef_opt_state
):
    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

    key, new_key = jax.random.split(key)
    (loss, individual_losses), grads = compute_loss(
        model,
        audio_frames=audio_frames,
        outputs_so_far=outputs_so_far,
        active_events=active_events,
        expected_next_output=next_output,
        key=key,
    )

    updates, update_opt_state = tx.update(grads, opt_state, model)
    update_model = eqx.apply_updates(model, updates)

    flat_update_model = jax.tree_util.tree_leaves(update_model)
    flat_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)

    return (loss, individual_losses), flat_update_model, flat_update_opt_state, new_key


@lru_cache(maxsize=1)
def load_test_set(testset_dir: Path, batch_size: int):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)
    chunks = chunked(sample_names, batch_size)

    batches = []
    for chunk in chunks:
        (
            frames,
            _,
            duration_per_frame,
            _frame_width,
        ) = AudioToMidiDatasetLoader.load_audio_frames_from_sample_name(testset_dir, chunk)
        midi_events = (
            AudioToMidiDatasetLoader.load_midi_events_frame_time_positions(
                testset_dir, chunk, duration_per_frame
            )
        )
        batches.append((frames, midi_events))
    return batches

@jax.jit
def compute_test_loss(
    model,
    key: jax.random.PRNGKey,
    audio_frames: Float[Array, "num_frames num_frame_data"],
    midi_events: Float[Array, "max_events 3"]
):
    size = midi_events.shape[0] - 1 # Minus one because we do not pick the last event
    mask = jnp.arange(size).reshape(1, size) <= jnp.arange(size).reshape(size, 1)
    mask = jnp.repeat(mask[..., None], repeats=3, axis=2)

    blank_event = jnp.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=jnp.int16)
    # Do not pick the last element with the full sequence, as there is nothing to predict
    event_prefixes = jnp.where(mask, midi_events[0:-1, ...], blank_event)
    active_events = jax.vmap(get_active_events)(event_prefixes)

    inference_keys = jax.random.split(key, num=event_prefixes.shape[0])
    midi_logits, _, _, position_probs, _, velocity_probs = jax.vmap(
        model, (None, 0, 0, 0)
    )(audio_frames, event_prefixes, active_events, inference_keys)

    losses, individual_losses = compute_loss_from_output(
        midi_logits,
        position_probs,
        velocity_probs,
        midi_events[1:, ...], # Skip the start of sequence event, but include the end of sequence
    )

    # Blank out the losses obtained when the prediction should result in a blank event
    actual_event_mask = midi_events[1:, 1] != BLANK_MIDI_EVENT
    losses = jnp.select([actual_event_mask], [losses], 0.0)
    return jnp.sum(losses) / jnp.count_nonzero(actual_event_mask)


def compute_testset_loss(model, testset_dir: Path, key: jax.random.PRNGKey, batch_size=32):
    batches = load_test_set(testset_dir, batch_size=batch_size)
    print("Loaded test set")

    test_loss = jnp.array([0.0], dtype=jnp.float32)
    for frames, midi_events in batches:
        test_loss_keys = jax.random.split(key, num=frames.shape[0])
        test_losses = jax.vmap(compute_test_loss, (None, 0, 0, 0))(model, test_loss_keys, frames, midi_events)
        test_loss += jnp.mean(test_losses)

    print("Finished evaluating test loss")
    return test_loss[0] / len(batches)


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
    idv_loss_sum = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        (audio_frames, seen_events, active_events, next_event) = jax.device_put(
            (batch["audio_frames"], batch["seen_events"], batch["active_events"], batch["next_event"]),
            batch_sharding,
        )

        (loss, individual_losses), flat_model, flat_state, key = compute_training_step(
            flat_model,
            audio_frames,
            seen_events,
            active_events,
            next_event,
            flat_state,
            key,
            tx,
            treedef_model,
            treedef_state,
        )
        step_end_time = time.time()

        if jnp.isnan(loss):
            print(f"Encountered NAN loss at step {step}. Stopping the training!")
            break

        if checkpoint_manager.should_save(step):
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            checkpoint_manager.save(step, args=ocp.args.StandardSave(model))

        loss_sum = loss_sum + loss
        idv_loss_sum = idv_loss_sum + individual_losses

        if step % print_every == 0:
            learning_rate = learning_rate_schedule(step)

            averaged_loss = (loss_sum / print_every)[0]
            averaged_individual_losses = idv_loss_sum / print_every

            if trainloss_csv is not None:
                trainloss_csv.writerow([step, averaged_loss, step_end_time - start_time, step * audio_frames.shape[0], learning_rate])
            print(f"Step {step}/{num_steps}, Loss: {averaged_loss}, LR = {learning_rate}, Idv.loss = {averaged_individual_losses}")

            loss_sum = jnp.array([0.0], dtype=jnp.float32)
            idv_loss_sum = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

        if step % testset_loss_every == 0 and testset_dir is not None:
            print("Evaluating test loss...")
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            eval_key, key = jax.random.split(key, num=2)
            testset_loss = compute_testset_loss(model, testset_dir, eval_key)
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
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v1")
    testset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v1-test-set")

    num_devices = len(jax.devices())

    batch_size = 16 * num_devices
    num_steps = 1000000
    learning_rate_schedule = create_learning_rate_schedule(2.5 * 1e-4, 1000, num_steps)

    checkpoint_every = 200
    checkpoints_to_keep = 3
    dataset_prefetch_count = 20
    dataset_num_workers = 2

    num_samples_to_load=10
    num_samples_to_maintain=batch_size * 10

    key = jax.random.PRNGKey(1234)
    model_init_key, training_key, dataset_loader_key = jax.random.split(key, num=3)

    print(f"Running on {num_devices} devices with an effective batch size of {batch_size}")

    # TODO: Enable dropout for training
    audio_to_midi = OutputSequenceGenerator(model_config, model_init_key)

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())
    audio_to_midi = jax.device_put(audio_to_midi, replicate_everywhere)

    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpoints_to_keep, save_interval_steps=checkpoint_every
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path, options=checkpoint_options
    )

    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is not None:
        print(f"Restoring saved model at step {step_to_restore}")
        audio_to_midi = checkpoint_manager.restore(
            step_to_restore,
            args=ocp.args.StandardRestore(audio_to_midi),
        )

    tx = optax.lion(learning_rate=learning_rate_schedule)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
    # The filtering is necessary to have the opt-state flattening working
    state = tx.init(eqx.filter(audio_to_midi, eqx.is_inexact_array))

    print("Setting up dataset loader...")
    dataset_loader = AudioToMidiDatasetLoader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        prefetch_count=dataset_prefetch_count,
        num_workers=dataset_num_workers,
        key=dataset_loader_key,
        num_samples_to_load=num_samples_to_load,
        num_samples_to_maintain=num_samples_to_maintain,
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
