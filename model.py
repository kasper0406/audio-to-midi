from functools import partial
from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

import position_encoding
from audio_to_midi_dataset import BLANK_MIDI_EVENT, BLANK_VELOCITY, MIDI_EVENT_VOCCAB_SIZE

model_config = {
    "frame_size": 4096,
    "max_frame_sequence_length": 98 + 1,
    "attention_size": 256,
    "intermediate_size": 512,
    "num_heads": 2,
    "num_layers": 5,
    "dropout_rate": 0.05,
    "midi_event_context_size": 15,
}


class FrameEmbedding(eqx.Module):
    """Takes frames from the audio samples and creates embeddings"""

    frame_embedder: eqx.nn.Linear
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
        self.frame_embedder = eqx.nn.Linear(self.frame_size, output_shape, key=key)
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
        frame_embeddings = jax.vmap(self.frame_embedder)(input_frames)
        position_embeddings = self.position_embeddings[0 : input_frames.shape[0]]
        combined = jax.vmap(self.layernorm)(frame_embeddings + position_embeddings)
        return self.dropout(combined, inference=not enable_dropout, key=key)


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

    node_embedding: eqx.nn.Embedding
    position_embeddings: Float[Array, "seq_len output_shape"]
    velocity_embedding: eqx.nn.Embedding
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
        node_key, velocity_key, event_type_key = jax.random.split(key, num=3)
        self.node_embedding = eqx.nn.Embedding(
            num_embeddings=self.voccab_size(), embedding_size=output_size, key=node_key
        )
        self.position_embeddings = position_encoding.for_input_frame(
            max_frame_sequence_length, output_size
        )
        self.velocity_embedding = eqx.nn.Embedding(
            num_embeddings=self.num_velocities(),
            embedding_size=output_size,
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
            Array, " 3"
        ],  # A numpy array of length 3. (midi event, position, velocity)
        event_type: Integer, # 0 for a seen event, 1 for a currently active event
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        midi_embedding = self.node_embedding(midi_event[1])
        position_embedding = self.position_embeddings[midi_event[0], :]
        velocity_embedding = self.velocity_embedding(midi_event[2])
        event_type_embedding = self.event_type_embedding(event_type)
        combined = self.layernorm(
            midi_embedding + position_embedding + velocity_embedding + event_type_embedding
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
        mask: Bool,
        enable_dropout: bool = True,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "attention_size"]:
        # Feed forward
        intermediate = self.attention_to_intermediate_proj(inputs)
        intermediate = jax.nn.gelu(
            intermediate
        )  # TODO: Consider trying a ReLU activation

        # Project back top attention space
        output = self.intermediate_to_attention_proj(intermediate)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Add residual and normalize the layers
        output = output * mask
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
        encoder_output: Optional[Float[Array, "seq_len attention_size"]],
        input_mask: Optional[
            Integer[Array, "seq_len"]
        ] = None,  # The mask on the attention inputs, notice we allow all encoder outputs to be attended to
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        mask = None
        if input_mask is not None:
            mask = self.make_attention_mask(
                input_mask,
                encoder_output.shape[0]
                if encoder_output is not None
                else inputs.shape[0],
                encoder_output is None,
            )

        attention_key, dropout_key = jax.random.split(key)

        # For encoding we use inputs as both the query, key and value in the attention mechanism
        # when decoding, we use the encoder outputs for the key and value
        result = self.attention(
            query=inputs,
            key_=encoder_output if encoder_output is not None else inputs,
            value=encoder_output if encoder_output is not None else inputs,
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
        input_mask: Integer[Array, " q_seq"],
        key_value_sequence_length: int,
        # TODO: Consider a nicer way of doing this
        mask_key_values: bool,  # If inputs are used as kv sequences, we need to mask those too!
    ) -> Float[Array, "q_seq kv_seq"]:
        """Create self-attention mask from sequence-level mask."""

        # In a non-self-attention layer where the kv arrays are the outputs of the encoder, we allow
        # the decoder to attend everywhere. Otherwise if the kv arrays are the seen inputs so far, we
        # mask out events not seen yet.
        kv_mask = jnp.ones(key_value_sequence_length, dtype=jnp.int32)
        if mask_key_values:
            kv_mask = input_mask

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

    self_attention_block: AttentionBlock
    encoder_attention_block: Optional[AttentionBlock]
    feed_forward_block: FeedForwardBlock

    def __init__(
        self,
        attention_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        allocate_encoder_attention: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self_attention_key, encoder_attention_key, feed_forward_key = jax.random.split(
            key, num=3
        )

        self.self_attention_block = AttentionBlock(
            attention_size=attention_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            key=self_attention_key,
        )

        self.encoder_attention_block = None
        if allocate_encoder_attention:
            self.encoder_attention_block = AttentionBlock(
                attention_size=attention_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=encoder_attention_key,
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
        encoder_output: Optional[Float[Array, "seq_len attention_size"]],
        mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        self_attention_key, encoder_attention_key, feed_forward_key = jax.random.split(
            key, num=3
        )

        output = self.self_attention_block(
            inputs, None, mask, enable_dropout, key=self_attention_key
        )

        if encoder_output is not None:
            output = self.encoder_attention_block(
                output, encoder_output, mask, enable_dropout, key=encoder_attention_key
            )

        seq_len = inputs.shape[0]
        feed_forward_keys = jax.random.split(feed_forward_key, num=seq_len)

        if mask is None:
            mask = jnp.ones((seq_len,), dtype=jnp.bool_)
        output = jax.vmap(self.feed_forward_block, in_axes=(0, 0, None, 0))(
            output, mask, enable_dropout, feed_forward_keys
        )
        return output


class Encoder(eqx.Module):
    """Use multiple TransformerLayer's to run the full encoding on the sequence"""

    frame_embedding: FrameEmbedding
    encoder_layers: list[TransformerLayer]
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        frame_size: int,
        max_frame_sequence_length: int,
        attention_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        num_layers: int,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        transformer_key, embedding_key = jax.random.split(key, num=2)

        self.frame_embedding = FrameEmbedding(
            output_shape=attention_size,
            frame_size=frame_size,
            max_frame_sequence_length=max_frame_sequence_length,
            key=embedding_key,
            dropout_rate=dropout_rate,
        )

        self.num_layers = num_layers
        layer_keys = jax.random.split(transformer_key, num_layers)
        def make_encoder_layer(layer_key):
            return TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                allocate_encoder_attention=False,
                key=layer_key,
            )
        self.encoder_layers = jax.vmap(make_encoder_layer)(layer_keys)

    def __call__(
        self,
        input_frames: Float[Array, "seq_len frame_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        transformer_key, frame_embedder_key = jax.random.split(key, num=2)

        embeddings = self.frame_embedding(
            input_frames, enable_dropout=enable_dropout, key=frame_embedder_key
        )

        dynamic_layers, static_layers = eqx.partition(self.encoder_layers, eqx.is_array)
        layer_keys = jax.random.split(transformer_key, num=self.num_layers)
        def compute_layer(current_state, current_dynamic_layer):
            idx, current_output = current_state
            layer_key = layer_keys[idx]
            layer = eqx.combine(current_dynamic_layer, static_layers)
            return (idx + 1, layer(
                inputs=current_output,
                mask=None, # There is no masking for the frame encoder
                encoder_output=None,
                enable_dropout=enable_dropout,
                key=layer_key,
            )), None

        (_, encoder_output), _ = jax.lax.scan(compute_layer, (0, embeddings), dynamic_layers)
        return encoder_output


class Decoder(eqx.Module):
    """Using encoder outputs, the generated input sequence, apply decoder transformers to
    compute a prob-dist over the next output in the sequence.
    """

    decoder_layers: List[TransformerLayer]
    midi_decoder_pooling: eqx.nn.Linear
    position_decoder_pooling: eqx.nn.Linear
    velocity_decoder_pooling: eqx.nn.Linear
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        num_layers: int,
        attention_size: int,
        intermediate_size: int,
        frame_seq_length: int,  # We need to know how many different input frames there are to be able to say when the event midi occours
        num_heads: int,
        dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        (
            transformer_key,
            midi_pooling_key,
            position_pooling_key,
            velocity_pooling_key,
        ) = jax.random.split(key, num=4)

        def make_decoder_layer(layer_key):
            return TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                allocate_encoder_attention=True,
                key=layer_key,
            )
        layer_keys = jax.random.split(transformer_key, num_layers)
        self.num_layers = num_layers
        self.decoder_layers = jax.vmap(make_decoder_layer)(layer_keys)

        self.midi_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=MidiVocabulary.voccab_size(),
            key=midi_pooling_key,
        )

        self.position_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=frame_seq_length,
            key=position_pooling_key,
        )

        self.velocity_decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=MidiVocabulary.num_velocities(),
            key=velocity_pooling_key,
        )

    # TODO: Consider a different attention_size for midi and frame embeddings
    def __call__(
        self,
        decoder_output: Float[
            Array, "midi_seq_len attention_size"
        ],  # The decoder output produced so far
        encoder_output: Float[Array, "frame_seq_len attention_size"],
        mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> (
        Float[Array, "midi_voccab_size"],
        Float[Array, "midi_voccab_size"],
        Float[Array, "frame_seq_len"],
        Float[Array, "frame_seq_len"],
    ):  # Probability distribution over the midi events
        (
            transformer_key,
            midi_decoder_key,
            position_decoder_key,
            velocity_decoder_key,
        ) = jax.random.split(key, num=4)

        dynamic_layers, static_layers = eqx.partition(self.decoder_layers, eqx.is_array)
        layer_keys = jax.random.split(transformer_key, num=self.num_layers)
        def compute_layer(current_state, current_dynamic_layer):
            idx, current_output = current_state
            layer_key = layer_keys[idx]
            layer = eqx.combine(current_dynamic_layer, static_layers)
            return (idx + 1, layer(
                inputs=current_output,
                mask=mask,
                encoder_output=encoder_output,
                enable_dropout=enable_dropout,
                key=layer_key,
            )), None

        (_, decoder_output), _ = jax.lax.scan(compute_layer, (0, decoder_output), dynamic_layers)

        # We take the first token in the last decoder layer to have information necessary for two
        # different pooling layers to extract both the midi event and the position
        # Notice: The first token must always be attendable by the mask! If not, the output of the token will
        #         not give any meaningful result.
        #         This padding is handled by our start of sequence token which is always pre-pended to the
        #         seen events, causing the mask for the first position to never be masked out.
        first_token_last_layer = decoder_output[0, :]
        midi_logits = self.midi_decoder_pooling(
            first_token_last_layer, key=midi_decoder_key
        )
        midi_probabilities = jax.nn.softmax(midi_logits)

        position_logits = self.position_decoder_pooling(
            first_token_last_layer, key=position_decoder_key
        )
        position_probabilities = jax.nn.softmax(position_logits)

        velocity_logits = self.velocity_decoder_pooling(
            first_token_last_layer, key=velocity_decoder_key
        )
        velocity_probabilities = jax.nn.softmax(velocity_logits)

        return (
            midi_logits,
            midi_probabilities,
            position_logits,
            position_probabilities,
            velocity_logits,
            velocity_probabilities,
        )


class OutputSequenceGenerator(eqx.Module):
    """
    1. Call the decoder on the current output sequence
    2. The output of the decoder is considered a probability distribution over possible outputs (according to the output spec)
    3. Select highest probability token until EOS token
    """

    encoder: Encoder
    decoder: Decoder

    midi_embedding: MidiVocabulary

    midi_event_context_size: int = eqx.field(static=True)

    def __init__(
        self,
        conf: Dict[str, any],
        key: Optional[jax.random.PRNGKey] = None,
    ):
        encoder_key, decoder_key, midi_embedding_key = jax.random.split(key, 3)

        self.encoder = Encoder(
            frame_size=conf["frame_size"],
            max_frame_sequence_length=conf["max_frame_sequence_length"],
            attention_size=conf["attention_size"],
            intermediate_size=conf["intermediate_size"],
            num_heads=conf["num_heads"],
            dropout_rate=conf["dropout_rate"],
            num_layers=conf["num_layers"],
            key=encoder_key,
        )

        self.decoder = Decoder(
            num_layers=conf["num_layers"],
            attention_size=conf["attention_size"],
            intermediate_size=conf["intermediate_size"],
            num_heads=conf["num_heads"],
            frame_seq_length=conf["max_frame_sequence_length"],
            dropout_rate=conf["dropout_rate"],
            key=decoder_key,
        )

        self.midi_embedding = MidiVocabulary(
            conf["attention_size"],
            conf["max_frame_sequence_length"],
            dropout_rate=conf["dropout_rate"],
            key=midi_embedding_key,
        )

        self.midi_event_context_size = conf["midi_event_context_size"]

    def __call__(
        self,
        input_frames: Float[Array, "frame_seq_len frame_size"],
        seen_events: Integer[
            Array, "midi_seq_len 3"  # Pair of (input_frame_position, midi_key, velocity)
        ],  # Index of midi event in the vocabulary
        active_events: Integer[Array, "midi_seq_len 3"],
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ) -> Float[
        Array, "midi_voccab_size"
    ]:  # Probability distribution over the midi events:
        seen_events = OutputSequenceGenerator._truncate_midi_event_to_context_size(
            seen_events, context_size=self.midi_event_context_size)
        active_events = OutputSequenceGenerator._truncate_midi_event_to_context_size(
            active_events, context_size=self.midi_event_context_size)
        return self.call_model(input_frames, seen_events, active_events, key, enable_dropout)

    @partial(jax.jit, static_argnames=["enable_dropout"])
    def call_model(
        self,
        input_frames: Float[Array, "frame_seq_len frame_size"],
        seen_events: Integer[
            Array, "midi_context_size 3"  # Pair of (input_frame_position, midi_key, velocity)
        ],  # Index of midi event in the vocabulary
        active_events: Integer[Array, "midi_context_size 3"],
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        encoder_key, decoder_key, midi_embedding_key_seen, midi_embedding_key_active = jax.random.split(key, num=4)

        encoder_output = self.encoder(
            input_frames=input_frames, enable_dropout=enable_dropout, key=encoder_key
        )

        midi_embeddings_seen = jax.vmap(
            partial(
                self.midi_embedding,
                event_type=0,
                enable_dropout=enable_dropout,
                key=midi_embedding_key_seen,
            )
        )(seen_events)

        midi_embeddings_active = jax.vmap(
            partial(
                self.midi_embedding,
                event_type=1,
                enable_dropout=enable_dropout,
                key=midi_embedding_key_seen,
            )
        )(active_events)

        all_midi_embeddings = jnp.concatenate([midi_embeddings_seen, midi_embeddings_active], axis=0)
        # Mask out all events with id of BLANK_MIDI_EVENT as those are just padding entries
        all_midi_events = jnp.concatenate([seen_events, active_events], axis=0, dtype=jnp.int16)
        mask = jnp.asarray(all_midi_events[:, 1] != BLANK_MIDI_EVENT, dtype=jnp.int32)

        midi_logits, midi_probs, position_logits, position_probs, velocity_logits, velocity_probs = self.decoder(
            decoder_output=all_midi_embeddings,
            encoder_output=encoder_output,
            mask=mask,
            enable_dropout=enable_dropout,
            key=decoder_key,
        )

        return midi_logits, midi_probs, position_logits, position_probs, velocity_logits, velocity_probs

    @partial(jax.jit, static_argnames=["context_size"])
    def _truncate_midi_event_to_context_size(generated_output: Integer[Array, "midi_seq_len 3"], context_size: int) -> Integer[Array, "midi_event_context_size 3"]:
        blank_event = jnp.array([0, BLANK_MIDI_EVENT, BLANK_VELOCITY], dtype=jnp.int16)

        num_events = jnp.count_nonzero(generated_output[:, 1] != BLANK_MIDI_EVENT, axis=0)
        offset = jnp.maximum(0, num_events - context_size)
        generated_output = jnp.roll(generated_output, shift=-offset, axis=0)
        events_to_keep = num_events - offset
        mask = jnp.repeat(jnp.arange(generated_output.shape[0])[:, None], repeats=3, axis=1) < events_to_keep
        generated_output = jnp.where(mask, generated_output, blank_event)
        generated_output = generated_output[0:context_size, ...]

        # Pad with necessary blank events to always reach the fixed context size
        padded_blanks = jnp.repeat(blank_event[None, ...], repeats=context_size, axis=0)
        generated_output = padded_blanks.at[:generated_output.shape[0], ...].set(generated_output)

        return generated_output
