from functools import partial
from typing import Dict, List, Tuple, Optional
import types
import json

import equinox as eqx
import jax
import jax.lib
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray, Key
import einops
from rope import calculate_rope, RopeFreqs

from audio_to_midi_dataset import MIDI_EVENT_VOCCAB_SIZE, get_data_prep_config

@jax.jit
def identity(arg):
    return arg

model_config = {
    "dims": [ 6 * (2 ** i) for i in range(8) ],
    "depths": [3] * 8,
    "cnn_hidden_expansion": 2.0,

    "num_transformer_layers": 5,
    "num_transformer_heads": 12,
    "attention_size": 64,
    "compressed_attention_q_size": 64,
    "compressed_attention_kv_size": 64,

    "transformer_dropout_rate": 0.1,
    "transformer_hidden_dim": 512,
    "transformer_hidden_expansion": 2.0,

    "sdd_rate": 0.05,
}


def get_model_metadata():
    metadata = {
        'model': model_config,
        'data_prep': get_data_prep_config(),
    }
    return metadata

def _split_key(key, num: int = 2):
    if key is None:
        return [ None ] * num
    else:
        return jax.random.split(key, num)

class LayerNorm(eqx.Module, strict=True):
    channels: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    weight: Float[Array, "channels"]
    bias: Float[Array, "channels"]

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        axis: int = 0,
    ):
        self.channels = channels
        self.eps = eps
        self.weight = jnp.ones((channels, ))
        self.bias = jnp.zeros((channels, ))
        self.axis = axis

    @jax.named_scope("kapper.LayerNorm")
    def __call__(
        self,
        x: Float[Array, "channels ..."],
        *,
        key: Key | None = None,
    ) -> Array | tuple[Array, eqx.nn.State]:
        orig_dtype = x.dtype
        with jax.numpy_dtype_promotion("standard"):
            dtype = jnp.result_type(x.dtype, jnp.float32)

        x = x.astype(dtype)
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        variance = jnp.var(x, axis=self.axis, keepdims=True)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv

        broadcast = [slice(None, None), *[jnp.newaxis] * (x.ndim - 1)]
        out = self.weight.astype(dtype)[*broadcast] * out  # pyright: ignore
        out = out + self.bias.astype(dtype)[*broadcast]  # pyright: ignore
        return out.astype(orig_dtype)

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


class Stem(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: LayerNorm

    def __init__(self, channels: int, kernel_size: int = 5, key: jax.random.PRNGKey = None):
        self.conv = eqx.nn.Conv1d(
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            key=key,
        )
        self.norm = LayerNorm(channels)
    
    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None):
        out = self.conv(x)
        return self.norm(out.astype(jnp.float32)).astype(out.dtype)

class Downsample(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: LayerNorm

    def __init__(self, in_channels: int, out_channels: int, key: Optional[jax.random.PRNGKey] = None):
        self.conv = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            key=key,
        )
        self.norm = LayerNorm(in_channels)
    
    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None):
        out = self.norm(x.astype(jnp.float32)).astype(x.dtype)
        return self.conv(out)

class GlobalResponseNorm(eqx.Module, strict=True):
    gamma: Array
    beta: Array

    def __init__(self):
        self.gamma = jnp.zeros((1,))
        self.beta = jnp.zeros((1,))

    def __call__(self, x):
        # print(f"x.shape in grn: {x.shape}")
        gx = jnp.sqrt(jnp.sum(x**2, axis=(1, ), keepdims=True) + 1e-6)
        # print(f"gx.shape in grn: {gx.shape}")
        nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + 1e-6)
        # jax.debug.print("gamma = {gamma}, beta = {beta}", gamma=self.gamma.value, beta=self.beta.value)
        return self.gamma * (x * nx) + self.beta + x

class Block(eqx.Module):
    depth_conv: eqx.nn.Conv1d
    point_conv_1: eqx.nn.Conv1d
    point_conv_2: eqx.nn.Conv1d
    stochastic_depth_dropout: StochasticDepthDropout
    norm: LayerNorm
    global_response_norm: GlobalResponseNorm

    def __init__(self, channels: int, hidden_dim: int, sdd_rate: float, kernel_size: int = 3, key: jax.random.PRNGKey = None):
        depth_conv_key, point_conv_1_key, point_conv_2_key = _split_key(key, 3)

        self.depth_conv = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            key=depth_conv_key,
            padding="SAME",
        )
        self.norm = LayerNorm(channels)

        self.point_conv_1 = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=2 * hidden_dim,
            kernel_size=1,
            key=point_conv_1_key,
        )

        self.point_conv_2 = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=channels,
            kernel_size=1,
            key=point_conv_2_key,
        )

        self.stochastic_depth_dropout = StochasticDepthDropout(sdd_rate)

        self.global_response_norm = GlobalResponseNorm()
    
    def __call__(self, x, enable_dropout: bool = False, key: Optional[jax.random.PRNGKey] = None):
        out = self.depth_conv(x)
        out = self.norm(out.astype(jnp.float32)).astype(out.dtype)
        out = self.point_conv_1(out)
        x1, x2 = jnp.split(out, 2, axis=0)
        out = jax.nn.gelu(x1) * x2
        # print(f"Out shape: {out.shape}")
        out = self.global_response_norm(out)
        # out = jax.nn.relu(out)
        out = self.point_conv_2(out)
        return self.stochastic_depth_dropout(out, inference=not enable_dropout, key=key) + x

class Decoder(eqx.Module):
    decoder_pooling: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        dim: int,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.decoder_pooling = eqx.nn.Linear(
            in_features=dim,
            out_features=MIDI_EVENT_VOCCAB_SIZE,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        x,
        key: Optional[jax.random.PRNGKey] = None,
    ):  # Probability distribution over the midi events
        output = jax.vmap(self.norm)(x.astype(jnp.float32)).astype(x.dtype)

        logits = jax.vmap(self.decoder_pooling)(output)
        probs = jax.nn.sigmoid(logits)

        return (
            logits,
            probs,
        )

class FeedForwardBlock(eqx.Module):
    attention_to_intermediate_proj: eqx.nn.Linear
    intermediate_to_attention_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_to_intermediate_key, intermediate_to_attention_key = _split_key(key, 2)
        self.attention_to_intermediate_proj = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=2 * intermediate_dim,  # x2 due to the glu-like activation
            key=attention_to_intermediate_key,
        )
        self.intermediate_to_attention_proj = eqx.nn.Linear(
            in_features=intermediate_dim,
            out_features=hidden_dim,
            key=intermediate_to_attention_key,
        )

        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "attention_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "attention_size"]:
        x = self.attention_to_intermediate_proj(inputs)
        x1, x2 = jnp.split(x, 2, axis=-1)
        h = jax.nn.gelu(x1) * x2  # GLU-like activation with gelu instead of silu

        output = self.intermediate_to_attention_proj(h)
        output = self.dropout(output, inference=not enable_dropout, key=key)
        return output


class SelfAttention(eqx.Module, strict=True):
    query_down_proj: eqx.nn.Linear | None = None
    query_up_proj: eqx.nn.Linear
    kv_down_proj: eqx.nn.Linear
    key_up_proj: eqx.nn.Linear
    value_up_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    local_attention_window: int | None = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int,
        head_dim: int,
        compressed_q_size: int,
        compressed_kv_size: int,
        dropout_rate: float = 0.0,
        inference: bool = False,
        local_attention_window: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        self.local_attention_window = local_attention_window
        q_down_key, q_up_key, kv_down_key, key_up_key, value_up_key, out_key = _split_key(key, 6)

        # if compressed_q_size != input_size:
        #     self.query_down_proj = eqx.nn.Linear(
        #         input_size,
        #         compressed_q_size,
        #         key=q_down_key,
        #         use_bias=False,
        #     )
        #     self.query_up_proj = eqx.nn.Linear(
        #         compressed_q_size,
        #         num_heads * head_dim,
        #         key=q_up_key,
        #         use_bias=False,
        #     )
        # else:
        self.query_up_proj = eqx.nn.Linear(
            input_size,
            num_heads * head_dim,
            key=q_up_key,
            use_bias=False,
        )


        self.kv_down_proj = eqx.nn.Linear(
            input_size,
            compressed_kv_size,
            key=kv_down_key,
            use_bias=False,
        )

        self.key_up_proj = eqx.nn.Linear(
            compressed_kv_size,
            num_heads * head_dim,
            key=key_up_key,
            use_bias=False,
        )

        self.value_up_proj = eqx.nn.Linear(
            compressed_kv_size,
            num_heads * head_dim,
            key=value_up_key,
            use_bias=False,
        )

        self.output_proj = eqx.nn.Linear(
            num_heads * head_dim,
            output_size,
            key=out_key,
            use_bias=False,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate, inference=inference)

        self.num_heads = num_heads

    @jax.named_scope("kapper.SelfAttention")
    def __call__(
        self,
        inputs: Float[Array, "q_seq hidden_dim"],
        rope_freqs: RopeFreqs,
        *,
        key: Optional[PRNGKeyArray] = None,
        enable_dropout: Optional[bool] = None,
    ) -> Float[Array, "q_seq o_size"]:
        query_seq_length, _ = inputs.shape

        c_q = inputs
        if self.query_down_proj:
            c_q = jax.vmap(self.query_down_proj)(inputs)
        query_heads = calculate_rope(self._project(self.query_up_proj, c_q), rope_freqs)

        c_kv = jax.vmap(self.kv_down_proj)(inputs)
        key_heads = calculate_rope(self._project(self.key_up_proj, c_kv), rope_freqs)
        value_heads = self._project(self.value_up_proj, c_kv)

        # TODO: Re-add dropout
        # print(f"Query heads shape: {query_heads.shape}")
        # attn_fn = partial(jax.nn.dot_product_attention, local_window_size=self.local_attention_window, implementation="cudnn")
        attn_fn = partial(jax.nn.dot_product_attention, local_window_size=self.local_attention_window)
        attn = attn_fn(query_heads, key_heads, value_heads)
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)


class TransformerLayer(eqx.Module):
    attention_norm: eqx.nn.LayerNorm
    attention_block: SelfAttention
    feed_forward_norm: eqx.nn.LayerNorm
    feed_forward_block: FeedForwardBlock

    def __init__(
        self,
        input_size: int,
        attention_size: int,
        compressed_attention_q_size: int,
        compressed_attention_kv_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        context_window: int | None = None,  # If context-window is specified, use local self-attention
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self_attention_key, feed_forward_key = _split_key(key, 2)
        self.attention_block = SelfAttention(
            input_size=input_size,
            output_size=input_size,
            head_dim=attention_size,
            num_heads=num_heads,
            compressed_q_size=compressed_attention_q_size,
            compressed_kv_size=compressed_attention_kv_size,
            dropout_rate=dropout_rate,
            key=self_attention_key,
            local_attention_window=context_window,
        )

        self.attention_norm = eqx.nn.LayerNorm(input_size)

        self.feed_forward_block = FeedForwardBlock(
            hidden_dim=input_size,
            intermediate_dim=intermediate_size,
            dropout_rate=dropout_rate,
            key=feed_forward_key,
        )
        self.feed_forward_norm = eqx.nn.LayerNorm(input_size)

    def __call__(
        self,
        inputs: Float[Array, "seq_len attention_size"],
        rope_freqs: RopeFreqs,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        encoder_attention_key, feed_forward_key = _split_key(key, num=2)

        r = self.attention_block(
            inputs=jax.vmap(self.attention_norm)(inputs.astype(jnp.float32)).astype(inputs.dtype),
            rope_freqs=rope_freqs,
            enable_dropout=enable_dropout,
            key=encoder_attention_key
        )

        h = inputs + r
        normalized_h = jax.vmap(self.feed_forward_norm)(h.astype(jnp.float32)).astype(h.dtype)
        if enable_dropout:
            feed_forward_keys = _split_key(feed_forward_key, num=h.shape[0])
            r = jax.vmap(self.feed_forward_block, in_axes=(0, None, 0))(
                normalized_h, enable_dropout, feed_forward_keys,
            )
        else:
            r = jax.vmap(self.feed_forward_block, in_axes=(0, None))(
                normalized_h, enable_dropout,
            )
        return h + r


class AlternatingLocalAndGlobalAttention(eqx.Module):
    attention_layers: List[TransformerLayer]

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        attention_size: int,
        intermediate_size: int,
        compressed_attention_q_size: int,
        compressed_attention_kv_size: int,
        dropout_rate: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        attention_windows = [20, 40, 60, None]
        keys = _split_key(key, len(attention_windows))

        self.attention_layers = []
        for attention_window, key in zip(attention_windows, keys):
            self.attention_layers.append(TransformerLayer(
                context_window=attention_window,
                input_size=input_size,
                attention_size=attention_size,
                compressed_attention_q_size=compressed_attention_q_size,
                compressed_attention_kv_size=compressed_attention_kv_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=key,
            ))
    
    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_dim"],
        rope_freqs: RopeFreqs,
        *,
        key: Optional[PRNGKeyArray] = None,
        enable_dropout: bool = False,
    ) -> Float[Array, "seq_len hidden_dim"]:
        x = inputs

        keys = _split_key(key, len(self.attention_layers))
        for layer, key in zip(self.attention_layers, keys):
            x = layer(x, rope_freqs, key=key, enable_dropout=enable_dropout)

        return x


class TransformerStack(eqx.Module):
    layers: list[TransformerLayer]
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        attention_size: int,
        compressed_attention_q_size: int,
        compressed_attention_kv_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.num_layers = num_layers

        def make_layer(layer_key):
            return AlternatingLocalAndGlobalAttention(
                input_size=input_size,
                attention_size=attention_size,
                compressed_attention_q_size=compressed_attention_q_size,
                compressed_attention_kv_size=compressed_attention_kv_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=layer_key,
            )

        keys = _split_key(key, num=num_layers)
        # self.layers = eqx.filter_vmap(make_layer)(keys)
        self.layers = []
        for layer_key in keys:
            self.layers.append(make_layer(layer_key))

    def __call__(
        self,
        inputs: Float[Array, "frames attention_size"],
        rope_freqs: RopeFreqs,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        # dynamic_layers, static_layer = eqx.partition(self.layers, eqx.is_inexact_array)

        # @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        # def f(x, spec):
        #     dynamic_layer, layer_key = spec
        #     layer = eqx.combine(dynamic_layer, static_layer)
        #     return layer(x, rope_freqs=rope_freqs, enable_dropout=enable_dropout, key=layer_key), None

        # if key is None:
        #     layer_keys = None
        # else:
        #     layer_keys = jnp.stack(_split_key(key, num=self.num_layers))
        # print(f"Dynamic layers: {dynamic_layers}")
        # print(f"Layer keys: {layer_keys}")
        # output, _ = jax.lax.scan(f, inputs, (dynamic_layers, layer_keys))

        output = inputs
        layer_keys = _split_key(key, num=len(self.layers)) if key is not None else [None] * len(self.layers)
        for layer, layer_key in zip(self.layers, layer_keys):
            output = layer(output, rope_freqs=rope_freqs, enable_dropout=enable_dropout, key=layer_key)

        return output


class OutputSequenceGenerator(eqx.Module):
    layers: list[eqx.nn.Sequential]
    norm: LayerNorm
    transformer_projection: eqx.nn.Linear | None = None
    transformer: TransformerStack
    decoder: Decoder

    recombination_rate: Array
    mutation_rate: Array

    def __init__(
        self,
        conf: Dict[str, any],
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.recombination_rate = jnp.array([1e-8], dtype=jnp.float32)
        self.mutation_rate = jnp.array([1e-7], dtype=jnp.float32)

        layers_key, decoder_key, transformer_projection_key, transformer_key = _split_key(key, 4)

        dims = conf["dims"]
        hidden_dims = [int(d * conf["cnn_hidden_expansion"]) for d in dims]
        depths = conf["depths"]

        self.layers = []

        layer_keys = _split_key(layers_key, len(dims))
        sdd_rates = jnp.linspace(0.0, conf["sdd_rate"], sum(depths))
        depth_count = 0
        for i, layer_key in zip(range(len(dims)), layer_keys):
            downsample_key, blocks_key = _split_key(layer_key, 2)

            downsample_layer = None
            if i == 0:
                downsample_layer = Stem(dims[0], key=downsample_key)
            else:
                downsample_layer = Downsample(dims[i - 1], dims[i], key=downsample_key)

            block_keys = _split_key(blocks_key, depths[i])

            self.layers.append(eqx.nn.Sequential([
                downsample_layer,
                *[
                    Block(dims[i], hidden_dims[i], sdd_rate=sdd_rates[depth_count + j], key=block_key)
                    for j, block_key in enumerate(block_keys)
                ],
            ]))
            depth_count += depths[i]

        self.norm = LayerNorm(dims[-1])
        # self.decoder = Decoder(dims[-1], key=decoder_key)

        transformer_hidden_dim = conf.get("transformer_hidden_dim", dims[-1])
        if transformer_hidden_dim != dims[-1]:
            self.transformer_projection = eqx.nn.Linear(
                dims[-1],
                transformer_hidden_dim,
                key=transformer_projection_key,
            )

        self.transformer = TransformerStack(
            input_size=transformer_hidden_dim,
            num_layers=conf["num_transformer_layers"],
            attention_size=conf["attention_size"],
            compressed_attention_q_size=conf["compressed_attention_q_size"],
            compressed_attention_kv_size=conf["compressed_attention_kv_size"],
            intermediate_size=int(transformer_hidden_dim * conf["transformer_hidden_expansion"]),
            num_heads=conf["num_transformer_heads"],
            dropout_rate=conf["transformer_dropout_rate"],
            key=transformer_key,
        )

        self.decoder = Decoder(transformer_hidden_dim, key=decoder_key)

    def __call__(
        self,
        samples: Float[Array, "frame_seq_len frame_size"],
        state,
        rope_freqs: RopeFreqs,
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        # samples = samples.astype(jnp.float16)

        print(f"Enable dropout? {enable_dropout}")
        print(f"Sample shape: {samples.shape}")
        resnext_key, transformer_key = _split_key(key, 2)
        layer_keys = _split_key(resnext_key, num=len(self.layers))

        # Compute ConvNext layers
        h = samples
        for layer, layer_key in zip(self.layers, layer_keys):
            h = layer(h, key=layer_key)
        h = self.norm(h.astype(jnp.float32)).astype(h.dtype)

        # # Compute Transformer layers
        h = jnp.transpose(h)
        if self.transformer_projection is not None:
            h = jax.vmap(self.transformer_projection)(h)
        h = self.transformer(h, rope_freqs=rope_freqs, enable_dropout=enable_dropout, key=transformer_key)

        # Decode the result
        logits, probs = self.decoder(h)
        return (logits, probs), state

    def predict(self, state, samples, rope_freqs: RopeFreqs):
        (logits, probs), _state = self(samples, state, rope_freqs, None)
        return logits, probs
