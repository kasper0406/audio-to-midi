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

from model import TransformerStack
from rope import calculate_rope

@jax.jit
def identity(arg):
    return arg

model_config = {
    "dims": [6, 10, 14, 24, 48, 96, 192, 384],
    "depths": [3, 3, 3, 3, 3, 3, 27, 3],

    "num_transformer_layers": 4,
    "num_transformer_heads": 1,
    "transformer_dropout_rate": 0.1,

    "sdd_rate": 0.1,
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
    norm: eqx.nn.LayerNorm

    def __init__(self, channels: int, kernel_size: int = 6, key: jax.random.PRNGKey = None):
        self.conv = eqx.nn.Conv1d(
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(channels)
    
    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None):
        out = self.conv(x)
        return jax.vmap(self.norm, in_axes=1, out_axes=1)(out)

class Downsample(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: eqx.nn.LayerNorm

    def __init__(self, in_channels: int, out_channels: int, key: Optional[jax.random.PRNGKey] = None):
        self.conv = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(in_channels)
    
    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None):
        out = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)
        return self.conv(out)

class Block(eqx.Module):
    depth_conv: eqx.nn.Conv1d
    point_conv_1: eqx.nn.Conv1d
    point_conv_2: eqx.nn.Conv1d
    stochastic_depth_dropout: StochasticDepthDropout
    norm: eqx.nn.LayerNorm
    gamma: Array

    def __init__(self, channels: int, hidden_dim: int, sdd_rate: float, kernel_size: int = 7, key: jax.random.PRNGKey = None):
        depth_conv_key, point_conv_1_key, point_conv_2_key = _split_key(key, 3)

        self.depth_conv = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            key=depth_conv_key,
            padding="SAME",
        )
        self.norm = eqx.nn.LayerNorm(channels)

        self.point_conv_1 = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=hidden_dim,
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

        layer_scale_value = 1e-6
        self.gamma = jnp.ones(channels) * layer_scale_value
    
    def __call__(self, x, enable_dropout: bool = False, key: Optional[jax.random.PRNGKey] = None):
        out = self.depth_conv(x)
        out = jax.vmap(self.norm, in_axes=1, out_axes=1)(out)
        out = self.point_conv_1(out)
        out = jax.nn.gelu(out)
        out = self.point_conv_2(out)
        out = self.gamma[:, None] * out  # Layer scale
        return self.stochastic_depth_dropout(out, inference=not enable_dropout, key=key) + x

class Decoder(eqx.Module):
    decoder_pooling: eqx.nn.Linear
    norm: eqx.nn.RMSNorm

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
        output = jax.vmap(self.norm)(x)

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
        attention_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_to_intermediate_key, intermediate_to_attention_key = _split_key(key, 2)
        self.attention_to_intermediate_proj = eqx.nn.Linear(
            in_features=attention_size,
            out_features=2 * intermediate_size,  # x2 due to the glu activation
            key=attention_to_intermediate_key,
        )
        self.intermediate_to_attention_proj = eqx.nn.Linear(
            in_features=intermediate_size,
            out_features=attention_size,
            key=intermediate_to_attention_key,
        )

        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "attention_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "attention_size"]:
        h = jax.nn.glu(self.attention_to_intermediate_proj(inputs))
        output = self.intermediate_to_attention_proj(h)
        output = self.dropout(output, inference=not enable_dropout, key=key)
        return output

class AttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention

    def __init__(
        self,
        attention_size: int,  # The attention size
        num_heads: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
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

    def __call__(
        self,
        inputs: Float[Array, "seq_len attention_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        # Self attention across the entire input sequence (nothing is masked)
        result = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            inference=not enable_dropout,
            key=key,
        )
        return result

class TransformerLayer(eqx.Module):
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
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self_attention_key, feed_forward_key = _split_key(key, 2)

        self.attention_block = AttentionBlock(
            attention_size=attention_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            key=self_attention_key,
        )
        self.attention_norm = eqx.nn.RMSNorm(attention_size)

        self.feed_forward_block = FeedForwardBlock(
            attention_size=attention_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=feed_forward_key,
        )
        self.feed_forward_norm = eqx.nn.RMSNorm(attention_size)

    def __call__(
        self,
        inputs: Float[Array, "seq_len attention_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        encoder_attention_key, feed_forward_key = _split_key(key, num=2)

        r = self.attention_block(
            inputs=jax.vmap(self.attention_norm)(inputs),
            enable_dropout=enable_dropout,
            key=encoder_attention_key
        )

        h = inputs + r
        feed_forward_keys = _split_key(feed_forward_key, num=h.shape[0])
        if enable_dropout:
            r = jax.vmap(self.feed_forward_block, in_axes=(0, None, 0))(
                jax.vmap(self.feed_forward_norm)(h), enable_dropout, feed_forward_keys,
            )
        else:
            r = jax.vmap(self.feed_forward_block, in_axes=(0, None))(
                jax.vmap(self.feed_forward_norm)(h), enable_dropout,
            )
        return h + r

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
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.num_layers = num_layers

        layer_keys = jax.random.split(key, num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=layer_key,
            ))

    def __call__(
        self,
        inputs: Float[Array, "frames attention_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        layer_keys = _split_key(key, num=self.num_layers)

        output = inputs
        for layer, layer_key in zip(self.layers, layer_keys):
            output = layer(
                inputs=output,
                enable_dropout=enable_dropout,
                key=layer_key,
            )

        return output

class PositionalEncoder(eqx.Module):
    embedding: eqx.nn.Embedding

    def __init__(self, max_length: int, attention_dim: int, key: Optional[jax.random.PRNGKey] = None):
        self.embedding = eqx.nn.Embedding(
            num_embeddings=max_length,
            embedding_size=attention_dim,
            key=key,
        )
    
    def __call__(self, input):
        # print(f"Input shape: {input.shape[0]}")
        return jax.vmap(self.embedding)(jnp.arange(input.shape[0]))

class OutputSequenceGenerator(eqx.Module):
    layers: list[eqx.nn.Sequential]
    transformer: TransformerStack
    pos_encoder: PositionalEncoder
    decoder: Decoder

    def __init__(
        self,
        conf: Dict[str, any],
        key: Optional[jax.random.PRNGKey] = None,
    ):
        layers_key, decoder_key, pos_encoding_key, transformer_key = _split_key(key, 4)

        dims = conf["dims"]
        hidden_dims = [d * 4 for d in dims]
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

        self.pos_encoder = PositionalEncoder(
            max_length=83,
            attention_dim=dims[-1],
            key=pos_encoding_key
        )

        self.transformer = TransformerStack(
            num_layers=conf["num_transformer_layers"],
            attention_size=dims[-1],
            intermediate_size=dims[-1] * 4,
            num_heads=conf["num_transformer_heads"],
            dropout_rate=conf["transformer_dropout_rate"],
            key=transformer_key,
        )

        self.decoder = Decoder(dims[-1], key=decoder_key)

    def __call__(
        self,
        samples: Float[Array, "frame_seq_len frame_size"],
        state,
        key: Optional[jax.random.PRNGKey] = None,
        enable_dropout: bool = False,
    ):
        print(f"Enable dropout? {enable_dropout}")
        print(f"Sample shape: {samples.shape}")
        resnext_key, transformer_key = _split_key(key, 2)
        layer_keys = _split_key(resnext_key, num=len(self.layers))

        # Compute ResNext layers
        h = samples
        for layer, layer_key in zip(self.layers, layer_keys):
            h = layer(h, key=layer_key)

        # Compute Transformer layers
        h = jnp.transpose(h)
        h = h + self.pos_encoder(h)
        h = self.transformer(h, enable_dropout=enable_dropout, key=transformer_key)

        # Decode the result
        logits, probs = self.decoder(h)
        return (logits, probs), state

    def predict(self, state, samples):
        (logits, probs), _state = self(samples, state, None)
        return logits, probs
