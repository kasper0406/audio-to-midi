import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray
import equinox as eqx
import optax
import functools

from model import OutputSequenceGenerator

config = {
    "frame_size": 1024,

    "attention_size": 128,
    "intermediate_size": 512,
    "num_heads": 2,
    "num_layers": 2,

    "dropout_rate": 0.05,
}

@eqx.filter_value_and_grad
def compute_loss(generator, audio_frames, outputs_so_far, expected_next_output, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    logits, _ = jax.vmap(generator, in_axes=(0, 0, 0))(
        audio_frames, outputs_so_far, batched_keys
    )
    print("Logits shape", logits.shape)
    print("expected_next_output shape", expected_next_output.shape)
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=expected_next_output)
    )

key = jax.random.PRNGKey(1234)
model_init_key, inference_key = jax.random.split(key)
audio_to_midi = OutputSequenceGenerator(config, model_init_key)

# _, next_token_probabilities = audio_to_midi(
#     input_frames=jnp.zeros(shape=(10, config["frame_size"]), dtype=jnp.float16),
#     generated_output=jnp.zeros(1, dtype=jnp.int16),
#     enable_dropout=False,
#     key=inference_key)
# print("Next token probs:", next_token_probabilities)

batch_size = 2
empty_audio_frames_batch = jnp.zeros(shape=(batch_size, 10, config["frame_size"]), dtype=jnp.float16) # Batch size 2, 10 frames
generated_sequence_batch = jnp.zeros(shape=(batch_size, 1), dtype=jnp.int16) # Put start of sequence token for each of the batches
expected_next_output_batch = jnp.ones(shape=(batch_size), dtype=jnp.int16) # For empty frames we expect no midi events

loss, grads = compute_loss(
    functools.partial(audio_to_midi, enable_dropout=False),
    audio_frames=empty_audio_frames_batch,
    outputs_so_far=generated_sequence_batch,
    expected_next_output=expected_next_output_batch,
    key=inference_key)
print("Loss:", loss)
