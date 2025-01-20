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
    "dims": [12, 16, 20, 24, 48, 96, 192, 384],
    "depths": [3, 3, 3, 3, 3, 3, 9, 3],

    "num_transformer_layers": 2,
    "transformer_num_heads": 1,
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

class OutputSequenceGenerator(eqx.Module):
    layers: list[eqx.nn.Sequential]
    transformer_stack: TransformerStack
    decoder: Decoder

    def __init__(
        self,
        conf: Dict[str, any],
        key: Optional[jax.random.PRNGKey] = None,
    ):
        layers_key, decoder_key, transformer_key = _split_key(key, 3)

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
        
        self.transformer_stack = TransformerStack(
            num_layers=conf["num_transformer_layers"],
            attention_size=dims[-1],
            intermediate_size=4 * dims[-1],
            num_heads=conf["transformer_num_heads"],
            dropout_rate=conf["transformer_dropout_rate"],
            stochastic_depth_dropout_rate=0.0,
            key=transformer_key,
        )

        self.decoder = Decoder(dims[-1], key=decoder_key)

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
        print(f"Sample shape: {samples.shape}")
        resnext_key, transformer_key = _split_key(key, 2)
        layer_keys = _split_key(resnext_key, num=len(self.layers))

        # Compute ResNext layers
        h = samples
        for layer, layer_key in zip(self.layers, layer_keys):
            h = layer(h, key=layer_key)
        
        # Compute Transformer layers
        h = jnp.transpose(h)
        mask = jnp.ones(h.shape[0], dtype=jnp.int32)
        h = self.transformer_stack(
            cos_freq=cos_freq,
            sin_freq=sin_freq,
            inputs=h,
            inputs_mask=mask,
            enable_dropout=enable_dropout,
            key=transformer_key,
        )

        # Decode the result
        logits, probs = self.decoder(h)
        return (logits, probs), state

    def predict(self, state, samples):
        (logits, probs), _state = self(samples, state, None)
        return logits, probs
