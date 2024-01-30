from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from dataset import audio_to_midi_dataset_loader
from model import OutputSequenceGenerator

config = {
    "frame_size": 232,  # 464,
    "max_frame_sequence_length": 256,
    "attention_size": 128,
    "intermediate_size": 512,
    "num_heads": 2,
    "num_layers": 2,
    "dropout_rate": 0.05,
}


@eqx.filter_jit
@eqx.filter_value_and_grad
def compute_loss(model, audio_frames, outputs_so_far, expected_next_output, key):
    batch_size = audio_frames.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    midi_logits, _, position_logits, _ = jax.vmap(model, in_axes=(0, 0, 0))(
        audio_frames, outputs_so_far, batched_keys
    )

    expected_next_midi, expected_next_position = expected_next_output
    midi_event_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=midi_logits, labels=expected_next_midi
    )
    # TODO: Consider using `softmax_cross_entropy` here and assign some probability density to nearby positions to give a bit of slack
    # TODO: Also consider if this can be represented in some other way
    position_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=position_logits, labels=expected_next_position
    )

    return jnp.mean(midi_event_loss + position_loss)


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
    data_loader,
    state: optax.OptState,
    num_steps: int = 10000,
    print_every: int = 1000,
    key: Optional[jax.random.PRNGKey] = None,
):
    losses = []
    for step, batch in zip(range(num_steps), data_loader):
        loss, model, state, key = compute_training_step(
            model,
            batch["audio_frames"],
            batch["seen_events"],
            batch["next_event"],
            state,
            key,
            tx,
        )

        losses.append(loss)
        if step % print_every == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss}")

    return model, state, losses


key = jax.random.PRNGKey(1234)
model_init_key, inference_key, training_key, dataset_loader_key = jax.random.split(
    key, num=4
)
audio_to_midi = OutputSequenceGenerator(config, model_init_key)

# _, midi_probs, _, position_probs = audio_to_midi(
#     input_frames=jnp.zeros(shape=(10, config["frame_size"]), dtype=jnp.float16),
#     generated_output=jnp.zeros(shape=(1, 2), dtype=jnp.int16),
#     enable_dropout=False,
#     key=inference_key,
# )
# print("Midi probs:", midi_probs)
# print("Position probs:", position_probs)

batch_size = 2
learning_rate = 1e-5

tx = optax.adam(learning_rate=learning_rate)
tx = optax.chain(optax.clip_by_global_norm(1.0), tx)  # TODO: Investigate clip by RMS
state = tx.init(audio_to_midi)

print("Setting up dataset loader...")
frame_loader = audio_to_midi_dataset_loader(dataset_loader_key, batch_size=batch_size)

# example_batch = next(frame_loader)
# print("Example batch", example_batch)
# loss, grads = compute_loss(
#     functools.partial(audio_to_midi, enable_dropout=False),
#     audio_frames=example_batch["audio_frames"],
#     outputs_so_far=example_batch["seen_events"],
#     expected_next_output=example_batch["next_event"],
#     key=inference_key,
# )
# print("Loss:", loss)

print("Starting training...")
trained_model, state, losses = train(
    audio_to_midi,
    frame_loader,
    state,
    num_steps=10,
    print_every=1,
    key=training_key,
)
