import random

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
from jaxtyping import Array, Float, PRNGKeyArray


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
    key: jax.random.PRNGKey,
    num_steps: int = 10000,
    print_every: int = 1000,
):
    losses = []
    for step, batch in zip(range(num_steps), data_loader):
        loss, model, state, key = compute_training_step(
            model,
            batch["position_encoding"],
            batch["position"],
            state,
            key,
            tx,
        )

        losses.append(loss)
        if step % print_every == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss}")

    return model, state, losses


def compute_position_encoding(
    position: int, model_dimension: int
) -> Float[Array, "model_dimension"]:
    progression_length = int(model_dimension / 2)
    base = jnp.zeros(progression_length) + 10_000
    exp = (jnp.arange(0, progression_length, 1.0) * 2) / model_dimension
    denominator = jnp.power(base, exp)
    numerator = jnp.zeros(progression_length) + position
    progression = numerator / denominator

    even_dim_encoding = jnp.sin(progression)
    odd_dim_encoding = jnp.cos(progression)

    combined = jnp.reshape(
        jnp.array([even_dim_encoding, odd_dim_encoding]), (model_dimension,), order="F"
    )

    return combined


class TrainingDatasetLoader(torch.utils.data.IterableDataset):
    def __init__(self, input_size: int, output_size: int):
        super(TrainingDatasetLoader).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def __iter__(self):
        while True:
            position = random.randint(0, self.output_size - 1)
            yield {
                "position_encoding": compute_position_encoding(
                    position, self.input_size
                ),
                "position": position,
            }

    @classmethod
    def collate(cls, samples):
        return {
            "position_encoding": jnp.array(
                [sample["position_encoding"] for sample in samples]
            ),
            "position": jnp.array([sample["position"] for sample in samples]),
        }


def main():
    input_size = 512  # Internal dimension of the network
    output_size = 1024  # Number of possible frames (this is probably too high)

    key = jax.random.PRNGKey(1234)
    model_init_key, inference_key, training_key, dataset_loader_key = jax.random.split(
        key, num=4
    )
    position_model = PositionEncodingRecovery(
        input_size=input_size, output_size=output_size, key=model_init_key
    )

    batch_size = 64
    learning_rate = 1e-3

    tx = optax.adam(learning_rate=learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
    state = tx.init(position_model)

    dataloader = torch.utils.data.DataLoader(
        TrainingDatasetLoader(input_size, output_size),
        batch_size=batch_size,
        num_workers=2,
        collate_fn=TrainingDatasetLoader.collate,
    )
    dataloader_iter = iter(dataloader)

    print("Starting training...")
    trained_model, state, losses = train(
        position_model,
        tx,
        dataloader_iter,
        state,
        print_every=50,
        key=training_key,
    )


if __name__ == "__main__":
    main()
