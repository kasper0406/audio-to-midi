from functools import partial
from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

import position_encoding
from audio_to_midi_dataset import BLANK_MIDI_EVENT, BLANK_VELOCITY, MIDI_EVENT_VOCCAB_SIZE, BLANK_DURATION, get_data_prep_config

model_config = {
    "frame_size": 1024,
    "max_frame_sequence_length": 200,
    "attention_size": 32,
    "intermediate_size": 32,
    "num_heads": 1,
    "num_layers": 2,
    "dropout_rate": 0.10,
    "midi_event_context_size": 30,
}

def get_model_metadata():
    return {
        'model': model_config,
        'data_prep': get_data_prep_config(),
    }


class FrameEmbedding(eqx.Module):
    """Takes frames from the audio samples and creates embeddings"""

    frame_embedder: eqx.nn.MLP
    frame_normalizer: eqx.nn.LayerNorm
    frame_size: int
    layernorm: eqx.nn.LayerNorm
    position_embeddings: Float[Array, "seq_len output_shape"]
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        # input_channels: int, # Always 1 for now, TODO consider 2 for stereo, 1 for mono
        output_shape: int,
        frame_size: int,  # Size of the processed audio frame
        max_frame_sequence_length: int,  # The maximum number of input frames
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        self.frame_size = frame_size
        self.frame_embedder = eqx.nn.MLP(
            in_size=self.frame_size,
            out_size=output_shape,
            width_size=self.frame_size,
            depth=0,
            key=key)
        self.frame_normalizer = eqx.nn.LayerNorm(shape=output_shape)
        self.layernorm = eqx.nn.LayerNorm(shape=output_shape)

        self.position_embeddings = position_encoding.for_input_frame(
            max_frame_sequence_length, output_shape
        )

        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        input_frames: Float[Array, "seq_len frame_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        position_embeddings = self.position_embeddings[0 : input_frames.shape[0]]
        frame_embeddings = jax.vmap(self.frame_embedder)(input_frames)
        frame_embeddings = jax.vmap(self.frame_normalizer)(frame_embeddings)
        combined = jax.vmap(self.layernorm)(frame_embeddings + position_embeddings)
        return self.dropout(combined, inference=not enable_dropout, key=key)

class NoteEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __init__(
        self,
        output_size: int,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.embedding = eqx.nn.Embedding(
            num_embeddings=MidiVocabulary.voccab_size(),
            embedding_size=output_size,
            key=key,
        )
    
    def __call__(
        self, 
        note: int,
    ):
        return self.embedding(note)

class VelocityEmbedding(eqx.Module):
    linear_velocity_embedding: eqx.nn.Linear

    def __init__(
        self,
        output_size: int,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        special_vel_key, linear_vel_key = jax.random.split(key, num=2)
        self.linear_velocity_embedding = eqx.nn.Linear(
            in_features=1,
            out_features=output_size,
            key=linear_vel_key,
        )

    def __call__(
        self, 
        velocity: int,
    ):
        scaled_vel = jnp.array([(velocity) / (MidiVocabulary.num_velocities())], dtype=jnp.float16)
        return self.linear_velocity_embedding(scaled_vel)

class DurationEmbedding(eqx.Module):
    special_duration_embedding: eqx.nn.Embedding
    position_transform: eqx.nn.Embedding
    position_embedding: Float[Array, "seq_len output_shape"]

    max_seq_len: int = eqx.field(static=True)

    def __init__(
        self,
        output_size: int,
        max_frame_sequence_length: int,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        special_duration_key, linear_duration_key = jax.random.split(key, num=2)

        self.special_duration_embedding = eqx.nn.Embedding(
            num_embeddings=1,
            embedding_size=output_size,
            key=special_duration_key
        )

        self.position_embedding = position_encoding.for_input_frame(
            max_frame_sequence_length, output_size
        )
        self.position_transform = eqx.nn.Linear(
            in_features=output_size,
            out_features=output_size,
            key=linear_duration_key
        )

        self.max_seq_len = max_frame_sequence_length

    def __call__(
        self, 
        duration: int,
    ):
        """
        One special duration:
        0: BLANK_DURATION
        """
        num_special = self.special_duration_embedding.num_embeddings

        def special_duration(special_duration):
            return self.special_duration_embedding(special_duration)
        def normal_duration(normal_duration):
            return self.position_transform(self.position_embedding[normal_duration, :])

        return jax.lax.cond(
            duration < num_special,
            special_duration,
            normal_duration,
            duration
        )

class MidiVocabulary(eqx.Module):
    """Takes midi events and creates an embedding of them:

    A midi event with velocity 0 is taken to mean a release of the midi note.

    Notice we put releases prior to attacks in terms of midi event, this is to hopefully have
    to maintain as little context as possible in the model.

    Enumarate in the following way:
      0: SEQUENCE_END
      1: SEQUENCE_START
      ...
    
    We want to provide the model with all currently played events, as well as a context window
    into what is has already predicted. This is achieved using the `event_type_embedding` embedding,
    which has two categories - one for active, and one for seen events.

    For a total voccab size of 2 + 88 = 90
    """

    @classmethod
    def voccab_size(cls):
        return MIDI_EVENT_VOCCAB_SIZE

    @classmethod
    def num_velocities(cls):
        return 10

    note_embedding: NoteEmbedding
    attack_time_embedding: Float[Array, "seq_len output_shape"]
    duration_embedding: DurationEmbedding
    velocity_embedding: VelocityEmbedding
    event_type_embedding: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        output_size: int,
        max_frame_sequence_length: int,
        dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        node_key, velocity_key, duration_key, event_type_key = jax.random.split(key, num=4)
        self.note_embedding = NoteEmbedding(
            output_size=output_size, key=node_key
        )
        self.attack_time_embedding = position_encoding.for_input_frame(
            max_frame_sequence_length, output_size
        )
        self.duration_embedding = DurationEmbedding(
            output_size=output_size,
            max_frame_sequence_length=max_frame_sequence_length,
            key=duration_key,
        )
        self.velocity_embedding = VelocityEmbedding(
            output_size=output_size,
            key=velocity_key,
        )
        self.event_type_embedding = eqx.nn.Embedding(
            num_embeddings=2, # Seen event = 0, or active event = 1
            embedding_size=output_size,
            key=event_type_key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=output_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        midi_event: Integer[
            Array, " 4"
        ],  # A numpy array of length 3. (position, note, velocity)
        event_type: Integer, # 0 for a seen event, 1 for a currently active event
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        note_embedding = self.note_embedding(midi_event[1])
        attack_time_embedding = self.attack_time_embedding[midi_event[0], :]
        duration_embedding = self.duration_embedding(midi_event[2])
        velocity_embedding = self.velocity_embedding(midi_event[3])
        event_type_embedding = self.event_type_embedding(event_type)
        combined = self.layernorm(
            note_embedding + attack_time_embedding + duration_embedding + velocity_embedding + event_type_embedding
        )
        return self.dropout(combined, inference=not enable_dropout, key=key)

class FeedForwardBlock(eqx.Module):
    """A signel feed forward transformer block.
    This applies to every output (count of them are attention_size) of the attention mechanism
        to introduce non-linearity.

    Does the following things:
        1. Project the attention result to an intermediate_size internal representation
        2. Apply an activation function
        3. Project back to the attention_size dimension using another linear transformation
    """

    attention_to_intermediate_proj: eqx.nn.Linear
    intermediate_to_attention_proj: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        attention_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_to_intermediate_key, intermediate_to_attention_key = jax.random.split(
            key
        )
        self.attention_to_intermediate_proj = eqx.nn.Linear(
            in_features=attention_size,
            out_features=intermediate_size,
            key=attention_to_intermediate_key,
        )
        self.intermediate_to_attention_proj = eqx.nn.Linear(
            in_features=intermediate_size,
            out_features=attention_size,
            key=intermediate_to_attention_key,
        )

        self.layernorm = eqx.nn.LayerNorm(shape=attention_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "attention_size"],
        enable_dropout: bool = True,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "attention_size"]:
        # Feed forward
        intermediate = self.attention_to_intermediate_proj(inputs)
        intermediate = jax.nn.gelu(
            intermediate
        )

        # Project back to attention space
        output = self.intermediate_to_attention_proj(intermediate)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Add residual and normalize the layers
        output += inputs
        output = self.layernorm(output)

        return output


class AttentionBlock(eqx.Module):
    """A single attention transformer block"""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        attention_size: int,  # The attention size
        num_heads: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=attention_size,  # Defaults for `value_size` and `output_size` automatically assumes `query_size`
            use_key_bias=True,
            use_output_bias=True,
            use_query_bias=True,
            use_value_bias=True,
            dropout_p=dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=attention_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "seq_len attention_size"],
        kv_context: Optional[Float[Array, "seq_len attention_size"]] = None,
        input_mask: Optional[Integer[Array, "seq_len"]] = None,
        kv_mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        if kv_context is None:
            kv_mask = input_mask
        kv_seq_len = kv_context.shape[0] if kv_context is not None else inputs.shape[0]
        mask = self.make_attention_mask(inputs.shape[0], kv_seq_len, input_mask, kv_mask)
        attention_key, dropout_key = jax.random.split(key)

        result = self.attention(
            query=inputs,
            key_=kv_context if kv_context is not None else inputs,
            value=kv_context if kv_context is not None else inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs  # residual
        result = jax.vmap(self.layernorm)(result)

        return result

    def make_attention_mask(
        self,
        input_sequence_length: int,
        kv_sequence_length: int,
        input_mask: Optional[Integer[Array, "seq_len"]] = None,
        kv_mask: Optional[Integer[Array, "seq_len"]] = None,
    ) -> Float[Array, "q_seq kv_seq"]:
        """Create self-attention mask from sequence-level mask."""

        if input_mask is None:
            input_mask = jnp.ones(input_sequence_length, dtype=jnp.int32)
        if kv_mask is None:
            kv_mask = jnp.ones(kv_sequence_length, dtype=jnp.int32)

        mask = jnp.multiply(
            jnp.expand_dims(input_mask, axis=-1),
            jnp.expand_dims(kv_mask, axis=-2),
        )
        return mask.astype(jnp.bool_)


class TransformerLayer(eqx.Module):
    """Combines:
    1. Multi-Head self attention mechanism, to decide what in the input audio frames to attend to
    2. Feed-Forward NN to process the attention output in a non-linear fashion
    """

    attention_block: AttentionBlock
    feed_forward_block: FeedForwardBlock

    def __init__(
        self,
        attention_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self_attention_key, encoder_attention_key, feed_forward_key = jax.random.split(
            key, num=3
        )

        self.attention_block = AttentionBlock(
            attention_size=attention_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            key=self_attention_key,
        )

        self.feed_forward_block = FeedForwardBlock(
            attention_size=attention_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=feed_forward_key,
        )

    def __call__(
        self,
        inputs: Float[Array, "seq_len attention_size"],
        input_mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        self_attention_key, encoder_attention_key, feed_forward_key = jax.random.split(
            key, num=3
        )

        output = self.attention_block(
            inputs, input_mask=input_mask, enable_dropout=enable_dropout, key=encoder_attention_key
        )

        feed_forward_keys = jax.random.split(feed_forward_key, num=inputs.shape[0])
        output = jax.vmap(self.feed_forward_block, in_axes=(0, None, 0))(
            output, enable_dropout, feed_forward_keys
        )
        return output

class TransformerStack(eqx.Module):

    layers: list[(TransformerLayer, TransformerLayer)]
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        num_layers: int,
        attention_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.num_layers = num_layers

        def make_transformer(layer_key):
            return TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=layer_key,
            )

        layer_keys = jax.random.split(key, num_layers)
        self.layers = jax.vmap(make_transformer)(layer_keys)

    def __call__(
        self,
        inputs: Float[Array, "frames attention_size"],
        inputs_mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        layer_keys = jax.random.split(key, num=self.num_layers)

        dynamic_layers, static_layers = eqx.partition(self.layers, eqx.is_array)

        def compute_layer(current_state, current_layer):
            idx, transformer_state = current_state
            leyer_key = layer_keys[idx]
            transformer_layer = eqx.combine(current_layer, static_layers)

            transformer_output = transformer_layer(
                inputs=transformer_state,
                input_mask=inputs_mask,
                enable_dropout=enable_dropout,
                key=leyer_key,
            )

            return (idx + 1, transformer_output), None

        (_, output), _ = jax.lax.scan(compute_layer, (0, inputs), dynamic_layers)

        return output

class Decoder(eqx.Module):
    """Using encoder outputs, the generated input sequence, apply decoder transformers to
    compute a prob-dist over the next output in the sequence.
    """

    midi_decoder_pooling: eqx.nn.Linear
    attack_time_decoder_pooling: eqx.nn.Linear
    duration_decoder_pooling: eqx.nn.Linear
    velocity_decoder_pooling: eqx.nn.Linear

    def __init__(
        self,
        attention_size: int,
        frame_seq_length: int,  # We need to know how many different input frames there are to be able to say when the event midi occours
        key: Optional[jax.random.PRNGKey] = None,
    ):
        (
            midi_pooling_key,
            position_pooling_key,
            duration_pooling_key,
            velocity_pooling_key,
        ) = jax.random.split(key, num=4)

        self.midi_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=MidiVocabulary.voccab_size(),
            key=midi_pooling_key,
        )

        self.attack_time_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=frame_seq_length,
            key=position_pooling_key,
        )

        self.duration_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=1,
            key=duration_pooling_key,
        )

        self.velocity_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=1,
            key=velocity_pooling_key,
        )

    # TODO: Consider a different attention_size for midi and frame embeddings
    def __call__(
        self,
        output: Float[Array, "attention_size"],
        key: Optional[jax.random.PRNGKey] = None,
    ):  # Probability distribution over the midi events
        (
            midi_decoder_key,
            attack_time_decoder_key,
            duration_decoder_key,
            velocity_decoder_key,
        ) = jax.random.split(key, num=4)

        midi_logits = self.midi_decoder_pooling(
            output, key=midi_decoder_key
        )
        midi_probabilities = jax.nn.softmax(midi_logits)

        attack_time_logits = self.attack_time_decoder_pooling(
            output, key=attack_time_decoder_key
        )
        attack_time_probabilities = jax.nn.softmax(attack_time_logits)

        duration = jax.nn.relu(self.duration_decoder_pooling(output, key=duration_decoder_key))
        velocity = jax.nn.relu(self.velocity_decoder_pooling(output, key=velocity_decoder_key))

        return (
            midi_logits,
            midi_probabilities,
            attack_time_logits,
            attack_time_probabilities,
            duration,
            velocity,
        )


class OutputSequenceGenerator(eqx.Module):
    """
    1. Call the decoder on the current output sequence
    2. The output of the decoder is considered a probability distribution over possible outputs (according to the output spec)
    3. Select highest probability token until EOS token
    """

    midi_embedding: MidiVocabulary
    frame_embedding: FrameEmbedding

    event_processor: TransformerStack
    decoder: Decoder
    dropout: eqx.nn.Dropout

    midi_event_context_size: int = eqx.field(static=True)

    def __init__(
        self,
        conf: Dict[str, any],
        key: Optional[jax.random.PRNGKey] = None,
    ):
        frame_processor_key, event_processor_key, frame_embedding_key, midi_embedding_key, decoder_key = jax.random.split(key, 5)

        self.event_processor = TransformerStack(
            num_layers=conf["num_layers"],
            attention_size=conf["attention_size"],
            intermediate_size=conf["intermediate_size"],
            num_heads=conf["num_heads"],
            dropout_rate=conf["dropout_rate"],
            key=event_processor_key,
        )

        self.frame_embedding = FrameEmbedding(
            output_shape=conf["attention_size"],
            frame_size=conf["frame_size"],
            max_frame_sequence_length=conf["max_frame_sequence_length"],
            key=frame_embedding_key,
            dropout_rate=conf["dropout_rate"],
        )

        self.midi_embedding = MidiVocabulary(
            conf["attention_size"],
            conf["max_frame_sequence_length"],
            dropout_rate=conf["dropout_rate"],
            key=midi_embedding_key,
        )

        self.decoder = Decoder(
            attention_size=conf["attention_size"],
            frame_seq_length=conf["max_frame_sequence_length"],
            key=decoder_key,
        )

        self.dropout = eqx.nn.Dropout(conf["dropout_rate"])

        self.midi_event_context_size = conf["midi_event_context_size"]

    def __call__(
        self,
        input_frames: Float[Array, "frame_seq_len frame_size"],
        seen_events: Integer[
            Array, "midi_seq_len 4"  # Pair of (input_frame_position, midi_key, velocity)
        ],  # Index of midi event in the vocabulary
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ) -> Float[
        Array, "midi_voccab_size"
    ]:  # Probability distribution over the midi events:
        seen_events = OutputSequenceGenerator._truncate_midi_event_to_context_size(
            seen_events, context_size=self.midi_event_context_size)
        return self.call_model(input_frames, seen_events, key, enable_dropout)

    @eqx.filter_jit
    def call_model(
        self,
        input_frames: Float[Array, "frame_seq_len frame_size"],
        seen_events: Integer[
            Array, "midi_context_size 4"  # Pair of (attack_time, midi_key, duration, velocity)
        ],  # Index of midi event in the vocabulary
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        event_processor_key, frame_embedding_key, midi_embedding_key_seen, midi_embedding_key_active, decoder_key, dropout_key = jax.random.split(key, num=6)

        frame_embeddings = self.frame_embedding(
            input_frames, enable_dropout=enable_dropout, key=frame_embedding_key
        )

        midi_embeddings = jax.vmap(
            partial(
                self.midi_embedding,
                event_type=0,
                enable_dropout=enable_dropout,
                key=midi_embedding_key_seen,
            )
        )(seen_events)

        event_mask = jnp.asarray(seen_events[:, 1] != BLANK_MIDI_EVENT, dtype=jnp.int8)

        all_inputs = jnp.concatenate([midi_embeddings, frame_embeddings], axis=0)
        total_mask = jnp.concatenate([event_mask, jnp.ones(frame_embeddings.shape[0], dtype=jnp.int8)], axis=0)

        output = self.event_processor(
            inputs=all_inputs,
            inputs_mask=total_mask,
            enable_dropout=enable_dropout,
            key=event_processor_key,
        )
        output = jnp.tanh(output[0, :])
        output = self.dropout(output, inference=not enable_dropout, key=dropout_key)

        return self.decoder(output, decoder_key)

    @partial(jax.jit, static_argnames=["context_size"])
    def _truncate_midi_event_to_context_size(
        generated_output: Integer[Array, "midi_seq_len 4"],
        context_size: int,
    ) -> Integer[Array, "midi_event_context_size 4"]:
        blank_event = jnp.array([0, BLANK_MIDI_EVENT, BLANK_DURATION, BLANK_VELOCITY], dtype=jnp.int16)

        num_events = jnp.count_nonzero(generated_output[:, 1] != BLANK_MIDI_EVENT, axis=0)
        offset = jnp.maximum(0, num_events - context_size)
        generated_output = jnp.roll(generated_output, shift=-offset, axis=0)
        events_to_keep = num_events - offset
        mask = jnp.repeat(jnp.arange(generated_output.shape[0])[:, None], repeats=4, axis=1) < events_to_keep
        generated_output = jnp.where(mask, generated_output, blank_event)
        generated_output = generated_output[0:context_size, ...]

        # Pad with necessary blank events to always reach the fixed context size
        padded_blanks = jnp.repeat(blank_event[None, ...], repeats=context_size, axis=0)
        generated_output = padded_blanks.at[:generated_output.shape[0], ...].set(generated_output)

        return generated_output
