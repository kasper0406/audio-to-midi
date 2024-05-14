from pathlib import Path

import tensorflow as tf
from jax.experimental import jax2tf
import equinox as eqx
import jax
import equinox.internal as eqxi

from infer import load_newest_checkpoint
from model import model_config

# TODO: Figure out a way to not hardcode this
NUM_FRAMES = 197
FRAME_SIZE = model_config["frame_size"]

def export_model_to_tf(model, state):
    module = tf.Module()
    predict_fn = jax2tf.convert(
        eqxi.finalise_fn(model.predict),
        polymorphic_shapes=["...", f"(b, 2, {NUM_FRAMES}, {FRAME_SIZE})"],
        with_gradient=False,
        enable_xla=False, # Disable XLA because TFlite and TFjs may have issues
    ) 

    @tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=(None, 2, NUM_FRAMES, FRAME_SIZE), dtype=tf.float32)])
    def predict(data):
        return predict_fn(state, data)
    module.predict = predict

    tf.saved_model.save(module, "./tf_export/")

if __name__ == "__main__":
    current_directory = Path(__file__).resolve().parent
    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    model, state = load_newest_checkpoint(checkpoint_path)

    export_model_to_tf(model, state)
