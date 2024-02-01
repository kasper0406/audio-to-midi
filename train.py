from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from audio_to_midi_dataset import AudioToMidiDatasetLoader
from model import OutputSequenceGenerator


@eqx.filter_jit
@eqx.filter_value_and_grad
def compute_loss(model, audio_frames, outputs_so_far, expected_next_output, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    midi_logits, _, position_logits, _ = jax.vmap(model, in_axes=(0, 0, 0))(
        audio_frames, outputs_so_far, batched_keys
    )

    expected_next_midi = expected_next_output[:, 0]
    midi_event_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=midi_logits, labels=expected_next_midi
    )
    # TODO: Consider using `softmax_cross_entropy` here and assign some probability density to nearby positions to give a bit of slack
    # TODO: Also consider if this can be represented in some other way
    expected_next_position = expected_next_output[:, 1]
    position_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=position_logits, labels=expected_next_position
    )

    # TODO: Fix the weight on the position loss so it is not hard-coded, but part of the config
    return jnp.mean(midi_event_loss + 0.3 * position_loss)


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


def train(
    model,
    tx,
    data_loader,
    state: optax.OptState,
    checkpoint_manager: ocp.CheckpointManager,
    num_steps: int = 10000,
    print_every: int = 1000,
    key: Optional[jax.random.PRNGKey] = None,
):
    losses = []
    start_step = (
        checkpoint_manager.latest_step() + 1
        if checkpoint_manager.latest_step() is not None
        else 0
    )

    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        loss, model, state, key = compute_training_step(
            model,
            batch["audio_frames"],
            batch["seen_events"],
            batch["next_event"],
            state,
            key,
            tx,
        )

        checkpoint_manager.save(step, args=ocp.args.StandardSave(model))

        losses.append(loss)
        if step % print_every == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss}")

    return model, state, losses


def main():
    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/v0")

    batch_size = 128
    learning_rate = 1e-3
    num_steps = 10000

    checkpoint_every = 50
    checkpoints_to_keep = 3
    dataset_prefetch_count = 200
    dataset_num_workers = 40

    model_config = {
        "frame_size": 232,  # 464,
        "max_frame_sequence_length": 256,
        "attention_size": 128,
        "intermediate_size": 512,
        "num_heads": 2,
        "num_layers": 2,
        "dropout_rate": 0.05,
    }

    key = jax.random.PRNGKey(1234)
    model_init_key, inference_key, training_key, dataset_loader_key = jax.random.split(
        key, num=4
    )

    # TODO: Enable dropout for training
    audio_to_midi = OutputSequenceGenerator(model_config, model_init_key)

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
        num_steps=num_steps,
        print_every=1,
        key=training_key,
    )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main()
