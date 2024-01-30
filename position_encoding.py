from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@partial(jax.jit, static_argnames=["model_dimension"])
def compute_single(
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


@partial(jax.jit, static_argnames=["batch_size", "input_size"])
def compute_batch(
    batch_size: int, input_size: int, output_size: int, key: jax.random.PRNGKey
) -> Float[Array, "batch_size model_dimension"]:
    positions = jax.random.randint(
        key, shape=(batch_size,), minval=0, maxval=output_size
    )
    encodings = jax.vmap(compute_single, (0, None))(positions, input_size)
    return jnp.array(positions), encodings


@partial(jax.jit, static_argnames=["position_count", "model_dimension"])
def for_input_frame(position_count: int, model_dimension: int):
    positions = jnp.arange(position_count)
    return jax.vmap(compute_single, (0, None))(positions, model_dimension)
