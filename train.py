from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from audio_to_midi_dataset import BLANK_MIDI_EVENT, AudioToMidiDatasetLoader
from infer import batch_infer
from model import OutputSequenceGenerator, model_config


@eqx.filter_jit
def single_position_loss(probs, expected):
    """
    Some kind of loss function that will promote having approximately correct positions

    Assumption: The value of the integer `expected` should be in the range 0 <= expected <= probs.shape[0]!
    """
    epsilon = 0.0000001
    variance = 3.0
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
def compute_loss_from_output(midi_logits, position_probs, expected_next_output):
    expected_next_midi = expected_next_output[:, 1]
    midi_event_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=midi_logits, labels=expected_next_midi
    )
    # TODO: Consider if this can be represented in some other way
    expected_next_position = expected_next_output[:, 0]
    position_loss = jax.vmap(single_position_loss, (0, 0))(
        position_probs, expected_next_position
    )

    # TODO: Fix the weight on the position loss so it is not hard-coded, but part of the config
    return jnp.mean(midi_event_loss + 0.2 * position_loss)


@eqx.filter_jit
@eqx.filter_value_and_grad
def compute_loss(model, audio_frames, outputs_so_far, expected_next_output, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    midi_logits, _, _, position_probs = jax.vmap(model, in_axes=(0, 0, 0))(
        audio_frames, outputs_so_far, batched_keys
    )

    return compute_loss_from_output(midi_logits, position_probs, expected_next_output)


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


def fake_evaluate_model(
    model, key: jax.random.PRNGKey, dataset_dir: Path, sample_size=64
):
    """
    TODO: The evaluation is a bit fake right now, as we do not put aside a validation set.
    However, as long as the model sturggles even with the in-sample data, it is fine

    It is a bit like a loss function, but for now it mainly serves for me to get some
    intuition using a different metric than the loss how the model is performing.
    """
    print("Starting model evaluation")

    # Pick a sublist of the samples and use those for evaluation
    sample_names = AudioToMidiDatasetLoader.load_sample_names(dataset_dir)[:sample_size]
    (
        frames,
        sample_rate,
        duration_per_frame,
    ) = AudioToMidiDatasetLoader.load_audio_frames(dataset_dir, sample_names)
    inferred_events, raw_outputs = batch_infer(model, key, frames)

    actual_events = AudioToMidiDatasetLoader.load_midi_events(dataset_dir, sample_names)

    min_dim = min(inferred_events.shape[1], actual_events.shape[1])
    # This is a bit weird, but just some kind of penalty for predicting the
    dim_penalty = jnp.sum(
        inferred_events[:, min_dim:, 1] != BLANK_MIDI_EVENT
    ) + jnp.sum(actual_events[:, min_dim:, 1] != BLANK_MIDI_EVENT)

    # For now just count the amount of correctly predicted events where a position is deemed correct
    # if it is within 5 frames
    difference = jnp.abs(
        inferred_events[:, :min_dim, :] - actual_events[:, :min_dim, :]
    )
    position_penalty = jnp.sum(difference[:, :, 0] > 5) + dim_penalty
    midi_penalty = jnp.sum(difference[:, :, 1] > 1) + dim_penalty
    print(
        f"Evaluation of model - position penalty = {position_penalty}, midi_penalty = {midi_penalty}"
    )


def train(
    model,
    tx,
    data_loader,
    state: optax.OptState,
    checkpoint_manager: ocp.CheckpointManager,
    device_mesh: [],
    num_steps: int = 10000,
    print_every: int = 1000,
    inference_every: int = 500,
    key: Optional[jax.random.PRNGKey] = None,
    model_evaluator=None,
):
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

        checkpoint_manager.save(step, args=ocp.args.StandardSave(model))

        losses.append(loss)
        if step % print_every == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss}")

        if step % inference_every == 0 and model_evaluator is not None:
            eval_key, key = jax.random.split(key, num=2)
            model_evaluator(model, eval_key)

    return model, state, losses


def main():
    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")

    num_devices = len(jax.devices())

    batch_size = 4 * num_devices
    learning_rate = 5 * 1e-4
    num_steps = 1000000

    checkpoint_every = 1000
    checkpoints_to_keep = 3
    dataset_prefetch_count = 10
    dataset_num_workers = 2

    key = jax.random.PRNGKey(1234)
    model_init_key, training_key, dataset_loader_key = jax.random.split(key, num=3)

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

    tx = optax.adam(learning_rate=learning_rate)
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
    audio_to_midi, state, losses = train(
        audio_to_midi,
        tx,
        dataset_loader_iter,
        state,
        checkpoint_manager,
        device_mesh=device_mesh,
        num_steps=num_steps,
        print_every=1,
        key=training_key,
        inference_every=checkpoint_every,
        # model_evaluator=partial(fake_evaluate_model, dataset_dir=dataset_dir),
    )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    with jax.profiler.trace("/tmp/jax-trace"):
        main()
