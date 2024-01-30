import queue
import threading
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, Float, PRNGKeyArray

import position_encoding


class PositionEncodingRecovery(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(
        self,
        input_size: int,
        output_size: int,
        key: PRNGKeyArray,
    ):
        self.linear = eqx.nn.Linear(input_size, output_size, key=key)

    def __call__(self, x: Float[Array, "input_size"]):
        logits = self.linear(x)
        probabilities = jax.nn.softmax(logits)
        return logits, probabilities


@eqx.filter_value_and_grad
def compute_loss(model, position_encoding, position, key):
    logits, _ = jax.vmap(model)(position_encoding)
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=position)
    )


def compute_training_step(model, position_encoding, position, opt_state, key, tx):
    key, new_key = jax.random.split(key)
    loss, grads = compute_loss(
        model,
        position_encoding=position_encoding,
        position=position,
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
    key: jax.random.PRNGKey,
    num_steps: int = 10000,
    print_every: int = 1000,
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
            batch["position_encoding"],
            batch["position"],
            state,
            key,
            tx,
        )

        checkpoint_manager.save(step, args=ocp.args.StandardSave(model))

        losses.append(loss)
        if step % print_every == 0:
            print(
                f"Step {step}/{num_steps}, Loss: {loss}, Latest checkpoint step: {checkpoint_manager.latest_step()}"
            )

    return model, state, losses


class TrainingDatasetLoader:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        batch_size: int,
        prefetch_count: int,
        num_workers: int,
        key: jax.random.PRNGKey,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.queue = queue.Queue(prefetch_count + 1)

        worker_keys = jax.random.split(key, num=num_workers)
        for worker_id in range(num_workers):
            # TODO: Consider closing these?
            threading.Thread(
                target=partial(self._generate_batch, key=worker_keys[worker_id]),
                daemon=True,
            ).start()

    def __iter__(self):
        while True:
            positions, encodings = self.queue.get()
            yield {
                "position_encoding": encodings,
                "position": positions,
            }

    def _generate_batch(self, key: jax.random.PRNGKey):
        while True:
            key, batch_key = jax.random.split(key, num=2)
            batch = position_encoding.compute_batch(
                self.batch_size, self.input_size, self.output_size, batch_key
            )
            self.queue.put(batch)


def main():
    input_size = 512  # Internal dimension of the network
    output_size = 1024  # Number of possible frames (this is probably too high)

    batch_size = 128
    learning_rate = 1e-4
    num_steps = 10000
    checkpoint_every = 50
    checkpoints_to_keep = 3

    path = "/Volumes/git/ml/models/audio-to-midi/positional_encoding_models/"
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpoints_to_keep, save_interval_steps=checkpoint_every
    )
    checkpoint_manager = ocp.CheckpointManager(path, options=checkpoint_options)

    key = jax.random.PRNGKey(1234)
    model_init_key, inference_key, training_key, dataset_loader_key = jax.random.split(
        key, num=4
    )
    position_model = PositionEncodingRecovery(
        input_size=input_size, output_size=output_size, key=model_init_key
    )

    tx = optax.adam(learning_rate=learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
    state = tx.init(position_model)

    dataloader = TrainingDatasetLoader(
        input_size,
        output_size,
        batch_size=batch_size,
        prefetch_count=10,
        num_workers=2,
        key=dataset_loader_key,
    )
    dataloader_iter = iter(dataloader)

    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is not None:
        print(f"Restoring saved model at step {step_to_restore}")
        position_model = checkpoint_manager.restore(
            step_to_restore,
            args=ocp.args.StandardRestore(position_model),
        )

    print("Starting training...")
    position_model, state, losses = train(
        position_model,
        tx,
        dataloader_iter,
        state,
        checkpoint_manager,
        print_every=25,
        num_steps=num_steps,
        key=training_key,
    )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
