import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from model import OutputSequenceGenerator

config = {
    "frame_size": 1024,

    "attention_size": 128,
    "intermediate_size": 512,
    "num_heads": 2,
    "num_layers": 2,

    "dropout_rate": 0.05,
}

key = jax.random.PRNGKey(1234)
model_init_key, inference_key = jax.random.split(key)
audio_to_midi = OutputSequenceGenerator(config, model_init_key)

next_token_probabilities = audio_to_midi(
    input_frames=jnp.zeros(shape=(10, config["frame_size"]), dtype=jnp.float16),
    generated_output=jnp.zeros(1, dtype=jnp.int16),
    enable_dropout=False,
    key=inference_key)

print("Next token probs:", next_token_probabilities)
