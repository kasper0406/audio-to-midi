from typing import Dict, List, Mapping, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer, PRNGKeyArray

class FrameEmbedding(eqx.Module):
    """Takes frames from the audio samples and creates embeddings"""

    linear: eqx.nn.Linear
    frame_size: int

    def __init__(
            self,
            # input_channels: int, # Always 1 for now, TODO consider 2 for stereo, 1 for mono
            output_shape: int,
            frame_size: int, # Size of the processed audio frame
            key: PRNGKeyArray
    ):
        self.frame_size = frame_size
        self.linear = eqx.nn.Linear(self.frame_size, output_shape, key=key)

    def __call__(self, x: Float[Array, "frame_size"]):
        output = self.linear(x)
        return output

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
        attention_to_intermediate_key, intermediate_to_attention_key = jax.random.split(key)
        self.attention_to_intermediate_proj = eqx.nn.Linear(
            in_features=attention_size, out_features=intermediate_size, key=attention_to_intermediate_key)
        self.intermediate_to_attention_proj = eqx.nn.Linear(
            in_features=intermediate_size, out_features=attention_size, key=intermediate_to_attention_key)
        
        self.layernorm = eqx.nn.LayerNorm(shape=attention_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)
    
    def __call__(
            self,
            inputs: Float[Array, "attention_size"],
            enable_dropout: bool = True,
            key: Optional[jax.random.PRNGKey] = None,
    ):
        # Feed forward
        intermediate = self.attention_to_intermediate_proj(inputs)
        intermediate = jax.nn.gelu(intermediate) # TODO: Consider trying a ReLU activation

        # Project back top attention space
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
            attention_size: int, # The attention size
            num_heads: int,
            dropout_rate: float,
            key: jax.random.PRNGKey,
    ):
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=attention_size, # Defaults for `value_size` and `output_size` automatically assumes `query_size`
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
            mask: Optional[Integer[Array, "seq_len"]], # The mask on the attention inputs to not leak future information
            enable_dropout: bool = False,
            key: Optional[jax.random.PRNGKey] = None
    ):
        # TODO

class TransformerLayer(eqx.Module):
    """Combines:
        1. Multi-Head self attention mechanism, to decide what in the input audio frames to attend to
        2. Feed-Forward NN to process the attention output in a non-linear fashion
    """

class Encoder(eqx.Module):
    """Use multiple TransformerLayer's to run the full encoding on the sequence"""

class Decoder(eqx.Module):
    """Combines:
        1. Masked multi-head attention: Help the decoder focus on the relevant part of the input
            (TODO: consider if looking slightly forward may be a help?)
        2. Feed-Forward NN
    """

class OutputSequenceGenerator(eqx.Module):
    """
        Output spec: [MidiToken] where MidiToken:
            - attack_frame: The frame count where the key was pressed (corresponding to the positional encoding of the audio frames)
            - release_frame: The frame count where the key was released (corresponding to the positional encoding of the audio frames)
            - key: The MIDI number of the key that was pressed (0->88)
            - TODO: OMIT FOR NOW: velocity: The velocity at which the key was pressed

        1. Call the decoder on the current output sequence
        2. The output of the decoder is considered a probability distribution over possible outputs (according to the output spec)
        3. Select highest probability token until EOS token
    """
