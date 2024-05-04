from functools import partial
from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray

import position_encoding
from audio_to_midi_dataset import MIDI_EVENT_VOCCAB_SIZE, get_data_prep_config

model_config = {
    "frame_size": 3715,
    "max_frame_sequence_length": 200,
    "attention_size": 32,
    "intermediate_size": 32,
    "num_heads": 1,
    "num_layers": 2,
    "dropout_rate": 0.10,
    "midi_event_context_size": 1,

    "internal_channels_1": 3,
    "time_kernel_1": 3,
    "time_stride_1": 1,
    "freq_kernel_1": 3,
    "freq_stride_1": 2,

    "internal_channels_2": 10,
    "time_kernel_2": 2,
    "time_stride_2": 1,
    "freq_kernel_2": 3,
    "freq_stride_2": 2,
}

def get_model_metadata():
    return {
        'model': model_config,
        'data_prep': get_data_prep_config(),
    }


class FrameEmbedding(eqx.Module):
    """Takes frames from the audio samples and creates embeddings"""

    frame_normalizer: eqx.nn.LayerNorm
    frame_size: int
    layernorm: eqx.nn.LayerNorm
    position_embeddings: Float[Array, "seq_len output_shape"]
    dropout: eqx.nn.Dropout
    position_embedder: eqx.nn.Linear
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(
        self,
        output_shape: int,
        frame_size: int,  # Size of the processed audio frame
        max_frame_sequence_length: int,  # The maximum number of input frames
        dropout_rate: float,
        key: PRNGKeyArray,
        input_channels=2,
    ):
        frame_key, pos_key, conv1_key, conv2_key, conv3_key = jax.random.split(key, num=5)

        self.frame_size = frame_size
        self.frame_normalizer = eqx.nn.LayerNorm(shape=output_shape)
        self.layernorm = eqx.nn.LayerNorm(shape=output_shape)

        self.position_embeddings = position_encoding.for_input_frame(
            max_frame_sequence_length, output_shape
        )
        self.position_embedder = eqx.nn.Linear(in_features=output_shape, out_features=output_shape, key=pos_key)

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.conv1 = eqx.nn.Conv2d(
            in_channels=input_channels,
            out_channels=model_config["internal_channels_1"],
            kernel_size=(model_config["time_kernel_1"], model_config["freq_kernel_1"]),
            stride=(model_config["time_stride_1"], model_config["freq_stride_1"]),
            key=conv1_key)

        self.conv2 = eqx.nn.Conv2d(
            in_channels=model_config["internal_channels_1"],
            out_channels=model_config["internal_channels_2"],
            kernel_size=(model_config["time_kernel_2"], model_config["freq_kernel_2"]),
            stride=(model_config["time_stride_2"], model_config["freq_stride_2"]),
            key=conv2_key)

        new_height = int(int(((self.frame_size - (model_config["freq_kernel_1"] - model_config["freq_stride_1"])) / model_config["freq_stride_1"]) - (model_config["freq_kernel_2"] - model_config["freq_stride_2"])) / model_config["freq_stride_2"])
        # print(f"New height {new_height}")
        self.conv3 = eqx.nn.Conv2d(
            in_channels=model_config["internal_channels_2"],
            out_channels=output_shape,
            kernel_size=(1, new_height),
            stride=(1, 1),
            key=conv3_key)

    def __call__(
        self,
        input_frames: Float[Array, "channels seq_len frame_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        # print(f"input_frames shape = {input_frames.shape}")
        c1 = self.conv1(input_frames)
        c1 = jax.nn.gelu(c1)
        c1 = jax.nn.normalize(c1)
        # print(f"c1 shape = {c1.shape}")

        c2 = self.conv2(c1)
        c2 = jax.nn.gelu(c2)
        c2 = jax.nn.normalize(c2)
        # print(f"c2 shape = {c2.shape}")

        frame_embeddings = jnp.transpose(jnp.squeeze((self.conv3(c2))))
        frame_embeddings = jax.nn.gelu(frame_embeddings)
        frame_embeddings = jax.nn.normalize(frame_embeddings)
        # print(f"frame embeddings shape = {frame_embeddings.shape}")

        position_embeddings = jax.vmap(self.position_embedder)(self.position_embeddings[0 : frame_embeddings.shape[0]])

        combined = jax.vmap(self.layernorm)(frame_embeddings + position_embeddings)
        return combined

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

    decoder_pooling: eqx.nn.Linear

    def __init__(
        self,
        attention_size: int,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.decoder_pooling = eqx.nn.Linear(
            in_features=attention_size,
            out_features=MIDI_EVENT_VOCCAB_SIZE,
            key=key,
        )

    def __call__(
        self,
        output: Float[Array, "attention_size"],
        key: Optional[jax.random.PRNGKey] = None,
    ):  # Probability distribution over the midi events

        logits = jax.vmap(self.decoder_pooling)(output)
        probs = jax.nn.sigmoid(logits)

        return (
            logits,
            probs,
        )

class OutputSequenceGenerator(eqx.Module):
    """
    1. Call the decoder on the current output sequence
    2. The output of the decoder is considered a probability distribution over possible outputs (according to the output spec)
    3. Select highest probability token until EOS token
    """

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

        self.decoder = Decoder(
            attention_size=conf["attention_size"],
            key=decoder_key,
        )

        self.dropout = eqx.nn.Dropout(conf["dropout_rate"])

        self.midi_event_context_size = conf["midi_event_context_size"]

    def __call__(
        self,
        input_frames: Float[Array, "frame_seq_len frame_size"],
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        event_processor_key, frame_embedding_key, decoder_key, dropout_key = jax.random.split(key, num=4)

        frame_embeddings = self.frame_embedding(
            input_frames, enable_dropout=enable_dropout, key=frame_embedding_key
        )
        mask = jnp.ones(frame_embeddings.shape[0], dtype=jnp.int8)

        output = self.event_processor(
            inputs=frame_embeddings,
            inputs_mask=mask,
            enable_dropout=enable_dropout,
            key=event_processor_key,
        )
        # output = jnp.tanh(output)
        output = self.dropout(output, inference=not enable_dropout, key=dropout_key)

        return self.decoder(output, decoder_key)
