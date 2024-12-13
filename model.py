from functools import partial
from typing import Dict, List, Optional
import types
import json

import equinox as eqx
import jax
import jax.lib
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray
import einops

import position_encoding
from audio_to_midi_dataset import MIDI_EVENT_VOCCAB_SIZE, get_data_prep_config

@jax.jit
def identity(arg):
    return arg

model_config = {
    "max_frame_sequence_length": 200,
    "attention_size": 128,
    "intermediate_size": 256,
    "num_heads": 4,
    "num_layers": 2,
    "dropout_rate": 0.15,
}

def serialize_function(obj):
    if isinstance(obj, types.FunctionType):
        return f"<function {obj.__name__}>"
    if isinstance(obj, jax.lib.xla_extension.PjitFunction):
        return f"<pjit_fn {obj.__name__}>"
    if isinstance(obj, jax.custom_jvp):
        return f"<custom_jvp {obj.__name__}>"
    if isinstance(obj, eqx.nn.PReLU):
        return "<eqx.nn.PReLU>"
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_model_metadata():
    metadata = {
        'model': model_config,
        'data_prep': get_data_prep_config(),
    }
    return json.dumps(metadata, default=serialize_function)

def _split_key(key, num: int = 2):
    if key is None:
        return [ None ] * num
    else:
        return jax.random.split(key, num)

class ConvSelfAttention(eqx.Module):
    to_qkv: eqx.nn.Conv
    to_output: eqx.nn.Conv
    num_heads: int
    input_dim: int
    norm: eqx.nn.BatchNorm

    def __init__(self, input_dim: int, internal_dim: int, num_heads: int, key: PRNGKeyArray | None):
        to_kqv_key, to_output_key = _split_key(key, 2)

        hidden_size = internal_dim * num_heads
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.to_qkv = eqx.nn.Conv1d(input_dim, 3 * hidden_size, kernel_size=1, key=to_kqv_key)
        self.to_output = eqx.nn.Conv1d(hidden_size, input_dim, kernel_size=1, key=to_output_key)
        self.norm = eqx.nn.BatchNorm(input_dim, axis_name="batch")

    def __call__(self, x, state, enable_dropout: bool = False):
        qkv = self.to_qkv(x)
        # print(f"qkv shape = {qkv.shape}")
        q, k, v = einops.rearrange(qkv, "(qkv heads c) l -> qkv heads l c", heads=self.num_heads, qkv=3)
        # print(f"q shape = {q.shape}, k shape = {k.shape}, v shape  {v.shape}")

        q = q / jnp.sqrt(q.shape[1])
        context = jnp.einsum("had,hbd -> hab",  q, k)
        context = jax.nn.softmax(context)
        # print(f"Context shape = {context.shape}")

        attention = jnp.einsum("hca,hcb -> hab", context, v)
        attention = einops.rearrange(attention, "heads l h -> (heads h) l")
        # print(f"Attention shape after = {attention.shape}")

        out = self.to_output(attention)
        out = out + x # residual
        out, state = self.norm(out, state, inference=not enable_dropout)

        return out, state

class ResidualConv(eqx.Module):
    scale_conv: eqx.nn.Conv
    conv: eqx.nn.Conv
    conv_first: eqx.nn.Conv
    activation_function: types.FunctionType
    batch_norm_0: eqx.nn.BatchNorm
    batch_norm_1: eqx.nn.BatchNorm
    batch_norm_2: eqx.nn.BatchNorm
    batch_norm_3: eqx.nn.BatchNorm
    dropout_1: eqx.nn.Dropout | None = None
    dropout_2: eqx.nn.Dropout | None = None
    shortcut: eqx.nn.Conv # Residual connections
    max_pool: eqx.nn.MaxPool1d | None = None
    avg_pool: eqx.nn.AvgPool1d | None = None
    attention: ConvSelfAttention | None = None
    position_encoding: jax.Array
    position_transform: eqx.nn.Conv1d

    def __init__(self, conv_inputs: int, channels: int, activation: types.FunctionType, max_pool: bool, avg_pool: bool, attention_dim: Optional[int], position_encoding: jax.Array, dropout_rate: float | None, key: PRNGKeyArray | None):
        scale_conv_key, conv_key, init_conv_key, shortcut_key, attention_key, pos_key = _split_key(key, 6)

        self.activation_function = activation
        out_channels = channels
        if activation == jax.nn.glu:
            out_channels = channels * 2
            self.activation_function = partial(jax.nn.glu, axis=0)
        
        kernel_size = 3
        self.conv_first = eqx.nn.Conv1d(
            in_channels=conv_inputs,
            out_channels=conv_inputs,
            kernel_size=kernel_size,
            padding=1,
            key=init_conv_key,
        )
        self.scale_conv = eqx.nn.Conv1d(
            in_channels=conv_inputs,
            out_channels=out_channels,
            kernel_size=(kernel_size + 1,),
            stride=(2,),
            padding=(1,),
            key=scale_conv_key
        )
        self.conv = eqx.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,),
            padding=(2,),
            key=conv_key
        )

        self.position_encoding = position_encoding
        self.position_transform = eqx.nn.Conv1d(
            in_channels=position_encoding.shape[1],
            out_channels=channels,
            kernel_size=1,
            key=pos_key
        )

        self.batch_norm_0 = eqx.nn.BatchNorm(input_size=conv_inputs, axis_name="batch")
        self.batch_norm_1 = eqx.nn.BatchNorm(input_size=out_channels, axis_name="batch")
        self.batch_norm_2 = eqx.nn.BatchNorm(input_size=channels, axis_name="batch")
        self.batch_norm_3 = eqx.nn.BatchNorm(input_size=channels, axis_name="batch")

        if dropout_rate is not None:
            self.dropout_1 = eqx.nn.Dropout(dropout_rate)
            self.dropout_2 = eqx.nn.Dropout(dropout_rate)
        
        self.shortcut = eqx.nn.Conv1d(
            in_channels=conv_inputs,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            key=shortcut_key
        )

        if max_pool:
            self.max_pool = eqx.nn.MaxPool1d(kernel_size=3, stride=2)
        if avg_pool:
            self.avg_pool = eqx.nn.AvgPool1d(kernel_size=3, stride=2)
        if attention_dim:
            self.attention = ConvSelfAttention(input_dim=channels, internal_dim=attention_dim, num_heads=4, key=attention_key)
    
    def __call__(self, x, state, enable_dropout: bool = False, key: PRNGKeyArray | None = None):
        x, state = self.batch_norm_0(x, state, inference=not enable_dropout)

        out = self.conv_first(x)
        out = self.activation_function(out)
        # out, state = self.batch_norm_0(out, state, inference=not enable_dropout)

        out = self.scale_conv(out)
        out = self.activation_function(out)
        # out, state = self.batch_norm_1(out, state, inference=not enable_dropout)

        # if self.dropout_1:
        #     out = self.dropout_1(out, inference=not enable_dropout, key=key)

        if self.attention:
            # Add positional encoding
            # print(f"Position encoding shape: {self.position_encoding[:x.shape[1], ...].shape}")
            pos_encoding = jnp.transpose(self.position_encoding[:out.shape[1], ...])
            out = out + self.position_transform(pos_encoding)
            # out = out + self.position_encoding[:out.shape[0], :out.shape[1]]

        conv2 = self.conv(out)
        conv2 = self.activation_function(out)
        # conv2, state = self.batch_norm_2(conv2, state, inference=not enable_dropout)
        if self.dropout_2:
            conv2 = self.dropout_2(conv2, inference=not enable_dropout, key=key)

        # Residual
        out = out + conv2 + self.shortcut(x)

        if self.max_pool:
            out = self.max_pool(out)
        if self.avg_pool:
            out = self.avg_pool(out)
        
        if self.attention:
            out, state = self.attention(out, state, enable_dropout=enable_dropout)
        
        return out, state

class FrameEmbedding(eqx.Module):
    """Takes frames from the audio samples and creates embeddings"""

    layernorm: eqx.nn.LayerNorm
    position_embeddings: Float[Array, "seq_len output_shape"]
    dropout: eqx.nn.Dropout
    position_embedder: eqx.nn.Linear
    layers: [eqx.Module]
    final_pooling: eqx.nn.Conv1d
    final_batch_norm: eqx.nn.BatchNorm

    def __init__(
        self,
        output_shape: int,
        max_frame_sequence_length: int,  # The maximum number of input frames
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        pos_key, conv_key, output_conv_key = jax.random.split(key, num=3)

        self.layernorm = eqx.nn.LayerNorm(shape=output_shape)

        max_attention_length = 2_000 # TODO: Do not hardcode this
        self.position_embeddings = position_encoding.for_input_frame(
            max_attention_length, output_shape
        )
        self.position_embedder = eqx.nn.Linear(in_features=output_shape, out_features=output_shape, key=pos_key)

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.layers = []
        num_layers = 8
        conv_keys = jax.random.split(conv_key, num=num_layers)

        max_num_features = 128
        for i, conv_key in zip(range(num_layers), conv_keys):
            out_channels = min(max_num_features, (2 ** (i + 3)))
            in_channels = min(max_num_features, (2 ** (i + 2)))
            if i == 0:
                in_channels = 2

            attention_dim = None
            if i > 3:
                attention_dim = out_channels

            self.layers.append(
                ResidualConv(
                    conv_inputs=in_channels,
                    channels=out_channels,
                    activation=jax.nn.relu,
                    max_pool=False,
                    avg_pool=False,
                    attention_dim=None,
                    position_encoding=self.position_embeddings,
                    dropout_rate=dropout_rate,
                    # dropout_rate=0.0,
                    key=conv_key,
                )
            )

        last_layer_features = min(max_num_features, 2 ** (num_layers + 2))
        self.final_pooling = eqx.nn.Conv1d(
            in_channels=last_layer_features,
            out_channels=output_shape,
            kernel_size=(3,),
            stride=(1,),
            key=output_conv_key
        )
        self.final_batch_norm = eqx.nn.BatchNorm(input_size=output_shape, axis_name="batch")

    def __call__(
        self,
        input_frames: Float[Array, "channels seq_len frame_size"],
        state,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        frame_embeddings = jnp.flip(input_frames, axis=1)
        layer_keys = _split_key(key, len(self.layers))
        for layer, layer_key in zip(self.layers, layer_keys):
            # print(f"Frame embedding shape: {frame_embeddings.shape}")
            frame_embeddings, state = layer(frame_embeddings, state, enable_dropout, layer_key)
        frame_embeddings = jnp.flip(frame_embeddings, axis=1)

        # frame_embeddings = self.final_pooling(frame_embeddings)
        # frame_embeddings, state = self.final_batch_norm(frame_embeddings, state, inference=not enable_dropout)
        # frame_embeddings = jax.nn.tanh(frame_embeddings)

        # Make the last layer fit what we need for the transformer
        frame_embeddings = jnp.transpose(jnp.squeeze((frame_embeddings)))
        # print(f"Frame embeddings shape: {frame_embeddings.shape}")

        # position_embeddings = jax.vmap(self.position_embedder)(self.position_embeddings[0 : frame_embeddings.shape[0]])
        # position_embeddings = jax.vmap(self.layernorm)(position_embeddings)
        position_embeddings = self.position_embeddings[0 : frame_embeddings.shape[0]]

        # combined = jax.vmap(self.layernorm)(frame_embeddings + position_embeddings)
        combined = frame_embeddings + position_embeddings
        out = self.dropout(combined, inference=not enable_dropout, key=key), state
        return out

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
        # self.layernorm = eqx.nn.LayerNorm(shape=intermediate_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "attention_size"],
        enable_dropout: bool = True,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "attention_size"]:
        # Feed forward
        intermediate = self.attention_to_intermediate_proj(inputs)
        intermediate = jax.nn.relu(intermediate)

        # Project back to attention space
        output = self.intermediate_to_attention_proj(intermediate)
        output = self.layernorm(output)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Add residual and normalize the layers
        output += inputs

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
        attention_key, dropout_key = _split_key(key)

        result = self.attention(
            query=inputs,
            key_=kv_context if kv_context is not None else inputs,
            value=kv_context if kv_context is not None else inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = result + inputs  # residual
        result = jax.vmap(self.layernorm)(result)

        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)

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
        return mask.astype(jnp.int32)


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
        self_attention_key, encoder_attention_key, feed_forward_key = _split_key(key, num=3)

        output = self.attention_block(
            inputs, input_mask=input_mask, enable_dropout=enable_dropout, key=encoder_attention_key
        )

        feed_forward_keys = _split_key(feed_forward_key, num=inputs.shape[0])
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
        layer_keys = _split_key(key, num=self.num_layers)

        dynamic_layers, static_layers = eqx.partition(self.layers, eqx.is_array)

        def compute_layer(current_state, current_layer):
            idx, transformer_state = current_state
            leyer_key = layer_keys[idx] if key is not None else None
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
            max_frame_sequence_length=conf["max_frame_sequence_length"],
            key=frame_embedding_key,
            dropout_rate=conf["dropout_rate"],
        )

        self.decoder = Decoder(
            attention_size=conf["attention_size"],
            key=decoder_key,
        )

        self.dropout = eqx.nn.Dropout(conf["dropout_rate"])

    def __call__(
        self,
        samples: Float[Array, "frame_seq_len frame_size"],
        state,
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        print(f"Enable dropout? {enable_dropout}")
        event_processor_key, frame_embedding_key, decoder_key, dropout_key = _split_key(key, num=4)
        print(f"Sample shape: {samples.shape}")

        frame_embeddings, state = self.frame_embedding(
            # OutputSequenceGenerator.__compress_samples(samples), state, enable_dropout=enable_dropout, key=frame_embedding_key
            samples, state, enable_dropout=enable_dropout, key=frame_embedding_key
        )
        mask = jnp.ones(frame_embeddings.shape[0], dtype=jnp.int32)

        output = self.event_processor(
            inputs=frame_embeddings,
            inputs_mask=mask,
            enable_dropout=enable_dropout,
            key=event_processor_key,
        )
        # output = jnp.tanh(output)
        # output = self.dropout(output, inference=not enable_dropout, key=dropout_key)

        logits, probs = self.decoder(output, decoder_key), state
        # logits, probs = self.decoder(frame_embeddings, decoder_key), state
        return logits, probs

    def predict(self, state, samples):
        (logits, probs), _state = self(samples, state, None)
        return logits, probs

    @eqx.filter_jit
    def __compress_samples(samples):
        def compress_samples(samples):
            mu = jnp.array(255.0, dtype=jnp.float16)
            return jnp.sign(samples) * jnp.log1p(mu * jnp.abs(samples)) / jnp.log1p(mu)
        return compress_samples(samples)
