from pathlib import Path

import tensorflow as tf
from jax.experimental import jax2tf
import equinox as eqx
import jax
import numpy as np
import equinox.internal as eqxi
import coremltools as ct

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
        # native_serialization=True,
    )

    @tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=(None, 2, NUM_FRAMES, FRAME_SIZE), dtype=tf.float32)])
    def predict(data):
        logits, probs = predict_fn(state, data)
        return {
            "logits": logits,
            "probs": probs,
        }
    module.predict = predict

    tf.saved_model.save(module, "./tf_export/",
        signatures={
            'predict': module.predict.get_concrete_function(),
        },
        options=tf.saved_model.SaveOptions(save_debug_info=True)
    )

def export_model_to_tf_lite():
    converter = tf.lite.TFLiteConverter.from_saved_model("./tf_export/")
    # converter.optimizations = [ tf.lite.Optimize.DEFAULT ]
    converter.target_spec = tf.lite.TargetSpec(
        # experimental_supported_backends=["GPU"], sad times... the model does not support this
        supported_ops=[
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ])
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

def export_model_to_coreml():
    coreml_model = ct.convert("./tf_export/",
        source='TensorFlow',
        debug=True,
        # inputs=[ct.TensorType(dtype=np.float16)],
        # outputs=[ct.TensorType(dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS17)
    coreml_model.save("audio2midi.mlpackage")

if __name__ == "__main__":
    current_directory = Path(__file__).resolve().parent
    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    model, state = load_newest_checkpoint(checkpoint_path)

    # export_model_to_tf(model, state)
    # export_model_to_tf_lite()
    export_model_to_coreml()
