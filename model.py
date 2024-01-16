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

class MidiVocabulary(eqx.Module):
    """Takes midi events and creates an embedding of them:
    
        Fixed representation. An event is described by: Type x Key
           where 
                Type: { ATTACK, RELEASE, END_OF_SEQUENCE }
                Key: [0..<88]
                # Velocity: Float
        
        Enumarate in the following way:
          0: EOS
          1: (ATTACK, 0)
          2: (RELEASE, 0)
          3: (ATTACK, 1)
          4: (RELEASE, 1)
          ...
        
        For a total voccab size of 1 + 2 * 88 = 177
    """

    @classmethod
    def voccab_size():
        return 177

    # TODO: Implement this

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
    ) -> Float[Array, "attention_size"]:
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
            encoder_output: Optional[Float[Array, "seq_len attention_size"]],
            enable_dropout: bool = False,
            key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "seq_len attention_size"]:
        if mask is not None:
            raise "Implement masking in the attention block!"
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
        result = result + inputs # residual
        result = jax.vmap(self.layernorm)(result)

        return result

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
            key: Optional[jax.random.PRNGKey],
    ):
        attention_key, feed_forward_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            attention_size=attention_size, num_heads=num_heads, dropout_rate=dropout_rate, key=attention_key)
        self.feed_forward_block = FeedForwardBlock(
            attention_size=attention_size, intermediate_size=intermediate_size, dropout_rate=dropout_rate, key=feed_forward_key)
        
    def __call__(
            self,
            inputs: Float[Array, "seq_len attention_size"],
            mask: Optional[Integer[Array, "seq_len"]],
            encoder_output: Optional[Float[Array, "seq_len attention_size"]],
            enable_dropout: bool = False,
            key=Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        attention_key, feed_forward_key = jax.random.split(key)
        
        attention_output = self.attention_block(inputs, encoder_output, mask, enable_dropout, key=attention_key)

        seq_len = inputs.shape[0]
        feed_forward_keys = jax.random.split(feed_forward_key, num=seq_len)
        output = jax.vmap(self.feed_forward_block, in_axes=(0, None, 0))(attention_output, enable_dropout, feed_forward_keys)
        return output

class Encoder(eqx.Module):
    """Use multiple TransformerLayer's to run the full encoding on the sequence"""

    frame_embedding: FrameEmbedding
    encoder_layers: list[TransformerLayer]

    def __init__(
            self,
            frame_size: int,
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
            key=embedding_key,
        )

        self.encoder_layers = []
        layer_keys = jax.random.split(transformer_key, num_layers)
        for layer_key in layer_keys:
            self.encoder_layers.append(TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=layer_key,
            ))

    def __call__(
            self,
            input_frames: Float[Array, "seq_len frame_size"],
            enable_dropout: bool = False,
            key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len attention_size"]:
        transformer_key = jax.random.split(key)

        embeddings = jax.vmap(self.frame_embedding)(input_frames)

        encoder_output = embeddings
        for layer in self.encoder_layers:
            transformer_key, layer_key = jax.random.split(transformer_key, num=2)
            encoder_output = layer(
                inputs=encoder_output,
                mask=None, # TODO: Implement masking
                encoder_output=encoder_output,
                enable_dropout=enable_dropout,
                key=layer_key,
            )

        return encoder_output

class Decoder(eqx.Module):
    """Using encoder outputs, the generated input sequence, apply decoder transformers to
       compute a prob-dist over the next output in the sequence.
    """

    decoder_layers: List[TransformerLayer]
    decoder_pooling: eqx.nn.Linear

    def __init__(
            self,
            num_layers: int,
            attention_size: int,
            intermediate_size: int,
            num_heads: int,
            dropout_rate: float,
            key: Optional[jax.random.PRNGKey] = None):
        transformer_key, pooling_key = self.jax.split(key, num=2)

        self.decoder_layers = []
        layer_keys = jax.random.split(transformer_key, num_layers)
        for layer_key in layer_keys:
            self.decoder_layers.append(TransformerLayer(
                attention_size=attention_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=layer_key
            ))

        self.decoder_pooling = eqx.nn.Linear(attention_size, MidiVocabulary.voccab_size(), key=pooling_key)

    # TODO: Distinguish between the two input/output sequence lengths
    # TODO: Consider a different attention_size for midi and frame embeddings
    def __call__(
            self,
            decoder_output: Float[Array, "midi_seq_len attention_size"], # The decoder output produced so far
            encoder_output: Float[Array, "frame_seq_len attention_size"],
            enable_dropout: bool = False,
            key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "midi_voccab_size"]: # Probability distribution over the midi events
        transformer_key, decoder_key = jax.random.split(key, 2)

        for layer in self.decoder_layers:
            transformer_key, layer_key = jax.random.split(transformer_key, 2)
            decoder_output = layer(
                inputs=decoder_output,
                mask=None, # TODO: Implement masking,
                encoder_output=encoder_output,
                enable_dropout=enable_dropout,
                key=layer_key,
                )

        pooled = self.decoder_pooling(decoder_output, key=decoder_key)
        pooled = jnp.softmax(pooled)

        return pooled

class OutputSequenceGenerator(eqx.Module):
    """
        1. Call the decoder on the current output sequence
        2. The output of the decoder is considered a probability distribution over possible outputs (according to the output spec)
        3. Select highest probability token until EOS token
    """

    encoder: Encoder
    decoder: Decoder

    def __init__(
            self,
            conf: Dict[str, any],
            key: Optional[jax.random.PRNGKey] = None,
    ):
        encoder_key, decoder_key = jax.random.split(key, 2)

        self.encoder = Encoder(
            frame_size=conf["frame_size"],
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
            dropout_rate=conf["dropout_rate"],
            key=decoder_key
        )

    def __call__(
            self,
            input_frames: Float[Array, "frame_seq_len frame_size"],
            generated_output: Integer[Array, "midi_seq_len"], # Index of midi event in the vocabulary
    ):
        encoder_key, decoder_key = jax.random.split(key, num=2)

        encoder_output = self.encoder(
            input_frames=input_frames,
            enable_dropout=enable_dropout,
            key=encoder_key
        )

        next_token_prob = self.decoder(
            decoder_output=TODO, # generate midi embeddings
            encoder_output=encoder_output,
            enable_dropout=enable_dropout,
            key=decoder_key,
        )

        return next_token_prob
