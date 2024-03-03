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

from audio_to_midi_dataset import BLANK_MIDI_EVENT, BLANK_VELOCITY, AudioToMidiDatasetLoader
from model import OutputSequenceGenerator, model_config


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
    # return optax.cosine_distance(probs, expectation)
    return optax.kl_divergence(jnp.log(probs + epsilon), expectation)

@eqx.filter_jit
def velocity_loss_fn(probs, expected, variance):
    epsilon = 0.0000001
    expectation = None
    if expected == 0:
        expectation = jnp.zeros(probs.shape[0])
        expectation = expectation.at[0].set(1.0)
    else:
        x = jnp.arange(probs.shape[0])
        expectation = -0.5 * jnp.square((x - expected) / variance)
        expectation = expectation.at[0].set(0.0)
        expectation = jnp.maximum(
            expectation, -10.0
        )  # Below values of -10 we will assume the exponential will give 0
        expectation = jnp.exp(expectation)

    expectation = expectation / jnp.sum(expectation)
    # jax.debug.print("Expectation: {expectation}", expectation=expectation)
    # return optax.cosine_distance(probs, expectation)
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
    return midi_event_loss + 0.3 * position_loss + 0.2 * velocity_loss


@eqx.filter_jit
@eqx.filter_value_and_grad
def compute_loss(model, audio_frames, outputs_so_far, expected_next_output, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    midi_logits, _, _, position_probs, _, velocity_probs = jax.vmap(
        model, in_axes=(0, 0, 0)
    )(audio_frames, outputs_so_far, batched_keys)

    return jnp.mean(compute_loss_from_output(
        midi_logits, position_probs, velocity_probs, expected_next_output
    ))


@eqx.filter_jit
def compute_training_step(
    model, audio_frames, outputs_so_far, next_output, opt_state, key, tx
):
    key, new_key = jax.random.split(key)
    loss, grads = compute_loss(
        model,
        audio_frames=audio_frames,
        outputs_so_far=outputs_so_far,
        expected_next_output=next_output,
        key=key,
    )

    updates, opt_state = tx.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state, new_key


@lru_cache(maxsize=1)
def load_test_set(testset_dir: Path):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)
    (
        frames,
        _,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames(testset_dir, sample_names)
    midi_events = (
        AudioToMidiDatasetLoader.load_midi_events_frame_time_positions(
            testset_dir, sample_names, duration_per_frame
        )
    )
    return frames, midi_events

@jax.jit
def compute_test_loss(
    model,
    key: jax.random.PRNGKey,
    audio_frames: Float[Array, "num_frames num_frame_data"],
    midi_events: Float[Array, "max_events 3"]
):
    size = midi_events.shape[0]
    mask = jnp.arange(size).reshape(1, size) <= jnp.arange(size).reshape(size, 1)
    mask = jnp.repeat(mask[..., None], repeats=3, axis=2)

    blank_event = jnp.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=jnp.int16)
    # Do not pick the last element with the full sequence, as there is nothing to predict
    event_prefixes = jnp.where(mask, midi_events, blank_event)[0:-1, ...]

    inference_keys = jax.random.split(key, num=event_prefixes.shape[0])
    midi_logits, _, _, position_probs, _, velocity_probs = jax.vmap(
        model, (None, 0, 0)
    )(audio_frames, event_prefixes, inference_keys)

    losses = compute_loss_from_output(
        midi_logits,
        position_probs,
        velocity_probs,
        midi_events[1:, ...], # Skip the start of sequence event, but include the end of sequence
    )

    # Blank out the losses obtained when the prediction should result in a blank event
    actual_event_mask = midi_events[1:, 1] != BLANK_MIDI_EVENT
    losses = jnp.select([actual_event_mask], [losses], 0.0)
    return jnp.sum(losses) / jnp.count_nonzero(actual_event_mask)


def compute_testset_loss(model, testset_dir: Path, key: jax.random.PRNGKey):
    frames, midi_events = load_test_set(testset_dir)

    print("Loaded test set")
    test_loss_keys = jax.random.split(key, num=frames.shape[0])
    test_losses = jax.vmap(compute_test_loss, (None, 0, 0, 0))(model, test_loss_keys, frames, midi_events)
    print("Finished evaluating test loss")
    return jnp.mean(test_losses)


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

    losses = []
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

    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        (audio_frames, seen_events, next_event) = jax.device_put(
            (batch["audio_frames"], batch["seen_events"], batch["next_event"]),
            batch_sharding,
        )

        loss, model, state, key = compute_training_step(
            model,
            audio_frames,
            seen_events,
            next_event,
            state,
            key,
            tx,
        )
        step_end_time = time.time()

        checkpoint_manager.save(step, args=ocp.args.StandardSave(model))

        losses.append(loss)
        if step % print_every == 0:
            learning_rate = learning_rate_schedule(step)
            if trainloss_csv is not None:
                trainloss_csv.writerow([step, loss, step_end_time - start_time, step * audio_frames.shape[0], learning_rate])
            print(f"Step {step}/{num_steps}, Loss: {loss}, LR = {learning_rate}")

        if step % testset_loss_every == 0 and testset_dir is not None:
            print("Evaluating test loss...")
            eval_key, key = jax.random.split(key, num=2)
            testset_loss = compute_testset_loss(model, testset_dir, eval_key)
            if testloss_csv is not None:
                testloss_csv.writerow([step, testset_loss, step_end_time - start_time, step * audio_frames.shape[0]])
            print(f"Test loss: {testset_loss}")

    return model, state, losses

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
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v1")
    testset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v1-test-set")

    num_devices = len(jax.devices())

    batch_size = 16 * num_devices
    num_steps = 1000000
    learning_rate_schedule = create_learning_rate_schedule(2.5 * 1e-4, 1000, num_steps)

    checkpoint_every = 200
    checkpoints_to_keep = 3
    dataset_prefetch_count = 0
    dataset_num_workers = 1

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

    tx = optax.adam(learning_rate=learning_rate_schedule)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0), tx
    )  # TODO: Investigate clip by RMS
    # state = tx.init(eqx.filter(audio_to_midi, eqx.is_inexact_array))
    state = tx.init(audio_to_midi)

    print("Setting up dataset loader...")
    dataset_loader = AudioToMidiDatasetLoader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        prefetch_count=dataset_prefetch_count,
        num_workers=dataset_num_workers,
        key=dataset_loader_key,
    )
    dataset_loader_iter = iter(dataset_loader)

    print("Starting training...")
    with open('train_loss.csv', mode='w', buffering=1) as trainloss_file:
        with open('test_loss.csv', mode='w', buffering=1) as testloss_file:
            trainloss_csv = csv.writer(trainloss_file)
            testloss_csv = csv.writer(testloss_file)

            audio_to_midi, state, losses = train(
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
                print_every=1,
                key=training_key,
                testset_loss_every=checkpoint_every,
            )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    with jax.profiler.trace("/tmp/jax-trace"):
        main()
