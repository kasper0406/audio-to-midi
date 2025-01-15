from functools import partial
from typing import Dict, List, Tuple, Optional
import types
import json

import equinox as eqx
import jax
import jax.lib
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray
import einops

from audio_to_midi_dataset import MIDI_EVENT_VOCCAB_SIZE, get_data_prep_config

from rope import calculate_rope

@jax.jit
def identity(arg):
    return arg

model_config = {
    "max_frame_sequence_length": 200,
    "attention_size": 64,
    "intermediate_size": 128,
    "num_heads": 2,
    "num_layers": 4,
    "dropout_rate": 0.20,
    "stochastic_depth_dropout_rate": 0.2,
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
    # return json.dumps(metadata, default=serialize_function)
    return metadata

def _split_key(key, num: int = 2):
    if key is None:
        return [ None ] * num
    else:
        return jax.random.split(key, num)


class StochasticDepthDropout(eqx.Module, strict=True):
    p: float
    inference: bool

    def __init__(
        self,
        p: float = 0.2,
        inference: bool = False,
    ):
        self.p = p
        self.inference = inference

    @jax.named_scope("kapper.StochasticDepthDropout")
    def __call__(
        self,
        x: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
    ) -> Array:
        if inference is None:
            inference = self.inference
        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        if inference:
            return x
        elif key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-deterministic mode."
            )
        else:
            rand = jax.random.uniform(key, shape=(1,))
            return jnp.where(rand < self.p, jnp.zeros_like(x), x)


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
    attention_to_intermediate_proj_2: eqx.nn.Linear
    intermediate_to_attention_proj: eqx.nn.Linear
    # layernorm: eqx.nn.RMSNorm
    dropout: eqx.nn.Dropout
    stochastic_depth_dropout: StochasticDepthDropout

    def __init__(
        self,
        attention_size: int,
        intermediate_size: int,
        dropout_rate: float,
        stochastic_depth_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_to_intermediate_key, attention_to_intermediate_key_2, intermediate_to_attention_key = jax.random.split(
            key, 3
        )
        self.attention_to_intermediate_proj = eqx.nn.Linear(
            in_features=attention_size,
            out_features=intermediate_size,
            key=attention_to_intermediate_key,
        )
        self.attention_to_intermediate_proj_2 = eqx.nn.Linear(
            in_features=attention_size,
            out_features=intermediate_size,
            key=attention_to_intermediate_key_2,
        )
        self.intermediate_to_attention_proj = eqx.nn.Linear(
            in_features=intermediate_size,
            out_features=attention_size,
            key=intermediate_to_attention_key,
        )

        # self.layernorm = eqx.nn.RMSNorm(shape=attention_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.stochastic_depth_dropout = StochasticDepthDropout(stochastic_depth_dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "attention_size"],
        enable_dropout: bool = True,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "attention_size"]:
        # Feed forward
        intermediate = jax.nn.sigmoid(self.attention_to_intermediate_proj(inputs)) * self.attention_to_intermediate_proj_2(inputs)

        dropout_key, sdd_key = _split_key(key, 2)

        # Project back to attention space
        output = self.intermediate_to_attention_proj(intermediate)
        output = self.dropout(output, inference=not enable_dropout, key=dropout_key)

        return self.stochastic_depth_dropout(output, inference=not enable_dropout, key=sdd_key) + inputs

class SqueezeExcite(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    prelu: eqx.nn.PReLU

    def __init__(self, channels, reduction=4, key=None):
        c1_key, c2_key = jax.random.split(key)
        hidden_dim = max(8, channels // reduction)
        self.fc1 = eqx.nn.Linear(channels, hidden_dim, key=c1_key)
        self.fc2 = eqx.nn.Linear(hidden_dim, channels, key=c2_key)
        self.prelu = eqx.nn.PReLU()

    def __call__(self, x):
        # x shape: (batch, channels, time)
        s = x.mean(axis=-1)  # global avg pool over time
        s = self.prelu(self.fc1(s))
        s = jax.nn.sigmoid(self.fc2(s))
        s = s[:, None]  # broadcast back over time dimension
        return x * s

class ScaleConv(eqx.Module):
    conv: eqx.nn.Conv
    shortcut_conv: eqx.nn.Conv
    downsample: eqx.nn.MaxPool1d
    norm: eqx.nn.RMSNorm | None = None
    sequeeze_excite: SqueezeExcite

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float | None, key: PRNGKeyArray | None):
        squeeze_excite_key, conv_key, shortcut_conv_key = _split_key(key, 3)

        self.conv = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,),
            stride=(1,),
            padding="SAME",
            key=conv_key,
        )
        self.shortcut_conv = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,),
            stride=(1,),
            padding="SAME",
            key=shortcut_conv_key,
        )
        self.downsample = eqx.nn.MaxPool1d(kernel_size=2, stride=2)
        self.sequeeze_excite = SqueezeExcite(out_channels, key=squeeze_excite_key)
        if in_channels > 8:
            self.norm = eqx.nn.RMSNorm(in_channels)
    
    def __call__(self, x, residual, state, enable_dropout: bool = False, key: PRNGKeyArray | None = None):
        out = x
        if self.norm:
            out = jax.vmap(self.norm, in_axes=1, out_axes=1)(out)

        out = self.sequeeze_excite(self.downsample(self.conv(out))) + self.downsample(self.shortcut_conv(residual))
        return out, out, state

class ResidualConv(eqx.Module):
    depthwise_conv_1: eqx.nn.Conv1d
    depthwise_conv_2: eqx.nn.Conv1d
    pointwise_conv_1: eqx.nn.Conv1d
    pointwise_conv_2: eqx.nn.Conv1d
    sequeeze_excite: SqueezeExcite
    alpha: float = eqx.field(static=True)
    norm: eqx.nn.RMSNorm | None = None
    stochastic_depth_dropout: StochasticDepthDropout

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout_rate: float | None, stochastic_depth_dropout_rate: float | None, key: PRNGKeyArray | None, alpha: float = 1.0):
        self.alpha = alpha

        depth_conv_1_key, depthwise_conv_2_key, pointwise_conv_1_key, pointwise_conv_2_key = _split_key(key, 4)
        self.depthwise_conv_1 = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size,),
            dilation=(dilation,),
            groups=channels,
            padding="SAME",
            key=depth_conv_1_key
        )
        self.depthwise_conv_2 = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size,),
            dilation=(dilation,),
            groups=channels,
            padding="SAME",
            key=depthwise_conv_2_key
        )

        self.pointwise_conv_1 = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1,),
            padding="SAME",
            key=pointwise_conv_1_key
        )
        self.pointwise_conv_2 = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1,),
            padding="SAME",
            key=pointwise_conv_2_key
        )

        self.sequeeze_excite = SqueezeExcite(channels, key=key)
        if channels > 8:
            self.norm = eqx.nn.RMSNorm(channels)
        
        self.stochastic_depth_dropout = StochasticDepthDropout(stochastic_depth_dropout_rate)
    
    def __call__(self, x, residual, state, enable_dropout: bool = False, key: PRNGKeyArray | None = None):
        out = x
        if self.norm:
            out = jax.vmap(self.norm, in_axes=1, out_axes=1)(out)

        # out = self.pointwise_conv_1(out)
        # out = self.depthwise_conv_1(out) * jax.nn.sigmoid(self.depthwise_conv_2(out))
        out = jax.nn.relu(self.depthwise_conv_1(out))
        out = self.pointwise_conv_2(out)
        out = self.sequeeze_excite(out)

        return self.stochastic_depth_dropout(self.alpha * out, inference=not enable_dropout, key=key) + x, residual, state

class ResidualConnection(eqx.Module):
    conv: eqx.nn.Conv1d
    pooling: eqx.nn.MaxPool1d

    def __init__(self, channels: int, residual_size: int, stride: int, key: PRNGKeyArray):
        self.conv = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=residual_size,
            kernel_size=(1,),
            padding="SAME",
            key=key,
        )
        self.pooling = eqx.nn.MaxPool1d(kernel_size=stride, stride=stride)

    def __call__(self, x, enable_dropout: bool = False, key: PRNGKeyArray | None = None):
        return self.pooling(self.conv(x))

class FrameEmbedding(eqx.Module):
    """Takes frames from the audio samples and creates embeddings"""

    layernorm: eqx.nn.RMSNorm
    dropout: eqx.nn.Dropout
    layers: List[Tuple[List[eqx.Module], ResidualConnection]]

    def __init__(
        self,
        output_shape: int,
        max_frame_sequence_length: int,  # The maximum number of input frames
        dropout_rate: float,
        stochastic_depth_dropout_rate: float,
        key: PRNGKeyArray,
    ):
        pos_key, conv_key, output_conv_key = jax.random.split(key, num=3)

        self.layernorm = eqx.nn.RMSNorm(shape=output_shape)
        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.layers = []
        num_layers = 10
        conv_keys = jax.random.split(conv_key, num=num_layers)

        attention_size = 64
        max_num_features = 34
        residual_size = (attention_size - max_num_features) // num_layers
        kernels_for_leyers = [
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
            [2, 3, 3, 5],
        ]
        upscale_factor = 0
        for i, conv_key, kernel_sizes in zip(range(num_layers), conv_keys, kernels_for_leyers):
            in_channels = min(max_num_features, (2 ** (i + 1 + upscale_factor)))
            if i == 0:
                in_channels = 2
            out_channels = min(max_num_features, (2 ** (i + 2 + upscale_factor)))

            block_layers = []
            main_res_conv_key, scale_conv_key, residual_conv_key = _split_key(conv_key, 3)
            res_conv_keys = _split_key(main_res_conv_key, len(kernel_sizes))
            for dilation, (kernel_size, res_conv_key) in enumerate(zip(kernel_sizes, res_conv_keys)):
                block_layers.append(
                    ResidualConv(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        # dilation=2 ** dilation,
                        dilation=1,
                        dropout_rate=dropout_rate,
                        stochastic_depth_dropout_rate=stochastic_depth_dropout_rate,
                        key=res_conv_key,
                    )
                )

            block_layers.append(
                ScaleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    key=scale_conv_key,
                )
            )

            residual_stride = 2 ** (num_layers - (i + 1))
            residual_connection = ResidualConnection(
                channels=out_channels,
                residual_size=residual_size,
                stride=residual_stride,
                key=residual_conv_key,
            )

            self.layers.append((block_layers, residual_connection))

    def __call__(
        self,
        input_frames: Float[Array, "channels seq_len frame_size"],
        state,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        frame_embeddings = input_frames
        block_keys = _split_key(key, len(self.layers))
        residual = frame_embeddings
        residuals = []
        for (layers, residual_block), block_key in zip(self.layers, block_keys):
            layer_keys = _split_key(key, len(layers))
            for layer, layer_key in zip(layers, layer_keys):
                # print(f"Frame embedding shape: {frame_embeddings.shape}")
                frame_embeddings, residual, state = layer(frame_embeddings, residual, state, enable_dropout, layer_key)
            residuals.append(residual_block(frame_embeddings))
            # print(f"Residual shape: {residuals[-1].shape}")

        # Add in the residuals to the frame embeddings
        residuals = [ residual[..., :frame_embeddings.shape[-1]] for residual in residuals ]
        frame_embeddings = jnp.concat([frame_embeddings] + residuals, axis=0)
        # print(f"Frame embeddings shape: {frame_embeddings.shape}")

        # Make the last layer fit what we need for the transformer
        frame_embeddings = jnp.transpose(jnp.squeeze((frame_embeddings)))

        return frame_embeddings, state


class AttentionBlock(eqx.Module):
    """A single attention transformer block"""

    attention: eqx.nn.MultiheadAttention
    # layernorm: eqx.nn.RMSNorm
    # dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)
    stochastic_depth_dropout: StochasticDepthDropout

    def __init__(
        self,
        attention_size: int,  # The attention size
        num_heads: int,
        dropout_rate: float,
        stochastic_depth_dropout_rate: float,
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
        # self.layernorm = eqx.nn.RMSNorm(shape=attention_size)
        # self.dropout = eqx.nn.Dropout(dropout_rate)

        self.stochastic_depth_dropout = StochasticDepthDropout(stochastic_depth_dropout_rate)

    def __call__(
        self,
        cos_freq: jax.Array,
        sin_freq: jax.Array,
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
        attention_key, dropout_key, sdd_key = _split_key(key, 3)

        kv_context = kv_context if kv_context is not None else inputs

        roped_inputs = calculate_rope(inputs, cos_freq, sin_freq)
        kv_context = calculate_rope(kv_context, cos_freq, sin_freq)

        result = self.attention(
            query=roped_inputs,
            key_=kv_context,
            value=kv_context,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        # result = jax.vmap(self.layernorm)(result)
        # result = self.dropout(result, inference=not enable_dropout, key=dropout_key)

        return self.stochastic_depth_dropout(result, inference=not enable_dropout, key=sdd_key) + inputs  # residual

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

    attention_norm: eqx.nn.RMSNorm
    attention_block: AttentionBlock
    feed_forward_norm: eqx.nn.RMSNorm
    feed_forward_block: FeedForwardBlock

    def __init__(
        self,
        attention_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        stochastic_depth_dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self_attention_key, encoder_attention_key, feed_forward_key = jax.random.split(
            key, num=3
        )

        self.attention_block = AttentionBlock(
            attention_size=attention_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            stochastic_depth_dropout_rate=stochastic_depth_dropout_rate,
            key=self_attention_key,
        )
        self.attention_norm = eqx.nn.RMSNorm(attention_size)

        self.feed_forward_block = FeedForwardBlock(
            attention_size=attention_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            stochastic_depth_dropout_rate=stochastic_depth_dropout_rate,
            key=feed_forward_key,
        )
        self.feed_forward_norm = eqx.nn.RMSNorm(attention_size)

    def __call__(
        self,
        cos_freq: jax.Array,
        sin_freq: jax.Array,
        inputs: Float[Array, "seq_len attention_size"],
        input_mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        self_attention_key, encoder_attention_key, feed_forward_key = _split_key(key, num=3)

        output = self.attention_block(
            cos_freq=cos_freq, sin_freq=sin_freq,
            inputs=jax.vmap(self.attention_norm)(inputs),
            input_mask=input_mask, enable_dropout=enable_dropout, key=encoder_attention_key
        )

        feed_forward_keys = _split_key(feed_forward_key, num=output.shape[0])
        if enable_dropout:
            output = jax.vmap(self.feed_forward_block, in_axes=(0, None, 0))(
                jax.vmap(self.feed_forward_norm)(output), enable_dropout, feed_forward_keys,
            )
        else:
            output = jax.vmap(self.feed_forward_block, in_axes=(0, None))(
                jax.vmap(self.feed_forward_norm)(output), enable_dropout,
            )
        return output

class TransformerStack(eqx.Module):

    layers: list[TransformerLayer]
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        num_layers: int,
        attention_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        stochastic_depth_dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.num_layers = num_layers

        def make_transformer(layer_key):
            return TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                stochastic_depth_dropout_rate=stochastic_depth_dropout_rate,
                key=layer_key,
            )

        layer_keys = jax.random.split(key, num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(make_transformer(layer_key))
        # self.layers = jax.vmap(make_transformer)(layer_keys)

    def __call__(
        self,
        cos_freq: jax.Array,
        sin_freq: jax.Array,
        inputs: Float[Array, "frames attention_size"],
        inputs_mask: Optional[Integer[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        layer_keys = _split_key(key, num=self.num_layers)

        # dynamic_layers, static_layers = eqx.partition(self.layers, eqx.is_array)

        # def compute_layer(current_state, current_layer):
            # idx, transformer_state = current_state
            # leyer_key = layer_keys[idx] if key is not None else None
            # transformer_layer = eqx.combine(current_layer, static_layers)

        def compute_layer(current_state, transformer_layer, layer_key):
            transformer_output = transformer_layer(
                cos_freq=cos_freq,
                sin_freq=sin_freq,
                inputs=current_state,
                input_mask=inputs_mask,
                enable_dropout=enable_dropout,
                key=layer_key,
            )

            # return (idx + 1, transformer_output), None
            return transformer_output

        # (_, output), _ = jax.lax.scan(compute_layer, (0, inputs), dynamic_layers)
        output = inputs
        for layer, layer_key in zip(self.layers, layer_keys):
            output = compute_layer(output, layer, layer_key)

        return output

class Decoder(eqx.Module):
    """Using encoder outputs, the generated input sequence, apply decoder transformers to
    compute a prob-dist over the next output in the sequence.
    """

    decoder_pooling: eqx.nn.Linear
    norm: eqx.nn.RMSNorm

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
        self.norm = eqx.nn.RMSNorm(attention_size)

    def __call__(
        self,
        output: Float[Array, "attention_size"],
        key: Optional[jax.random.PRNGKey] = None,
    ):  # Probability distribution over the midi events
        output = jax.vmap(self.norm)(output)

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
            stochastic_depth_dropout_rate=conf["stochastic_depth_dropout_rate"],
            key=event_processor_key,
        )

        self.frame_embedding = FrameEmbedding(
            output_shape=conf["attention_size"],
            max_frame_sequence_length=conf["max_frame_sequence_length"],
            key=frame_embedding_key,
            dropout_rate=conf["dropout_rate"],
            stochastic_depth_dropout_rate=conf["stochastic_depth_dropout_rate"],
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
        cos_freq: jax.Array,
        sin_freq: jax.Array,
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        print(f"Enable dropout? {enable_dropout}")
        event_processor_key, frame_embedding_key, decoder_key, dropout_key = _split_key(key, num=4)
        print(f"Sample shape: {samples.shape}")

        frame_embeddings, state = self.frame_embedding(
            samples, state, enable_dropout=enable_dropout, key=frame_embedding_key
        )
        mask = jnp.ones(frame_embeddings.shape[0], dtype=jnp.int32)

        output = self.event_processor(
            cos_freq=cos_freq,
            sin_freq=sin_freq,
            inputs=frame_embeddings,
            inputs_mask=mask,
            enable_dropout=enable_dropout,
            key=event_processor_key,
        )
        # output = jnp.tanh(output)
        output = self.dropout(output, inference=not enable_dropout, key=dropout_key)

        logits, probs = self.decoder(output, decoder_key), state
        return logits, probs

    def predict(self, state, samples):
        (logits, probs), _state = self(samples, state, None)
        return logits, probs
