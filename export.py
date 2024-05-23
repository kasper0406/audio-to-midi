from pathlib import Path

import tensorflow as tf
from jax.experimental import jax2tf
import equinox as eqx
import jax
import numpy as np
import equinox.internal as eqxi
import coremltools as ct
import coremltools.optimize as cto

from infer import load_newest_checkpoint
from model import model_config
from audio_to_midi_dataset import AudioToMidiDatasetLoader

# TODO: Figure out a way to not hardcode this
SAMPLE_RATE = 8000.0
DURATION = 2.0

TF_MODEL_PATH = "./tf_export/"
TF_PREPARE_PATH = "./tf_prepare/"

def export_model_to_tf(model, state):
    input_sample_count = int(DURATION * SAMPLE_RATE)

    module = tf.Module()
    predict_fn = jax2tf.convert(
        eqxi.finalise_fn(model.predict),
        polymorphic_shapes=["...", f"(2, {input_sample_count})"],
        # with_gradient=False, We can not convert without gradients as it generates an PreventGradient op that coremltools does not know about
        enable_xla=False,
        # enable_xla=True, # coremltools does no support Xla :(
        # native_serialization=True,
        # native_serialization_disabled_checks=[
        #     jax2tf.DisabledSafetyCheck.shape_assertions() # Required for coremltools to convert the resulting TF model
        # ]
    )

    @tf.function(
        autograph=False,
        input_signature=[tf.TensorSpec(shape=(2, input_sample_count), dtype=tf.float32)],
        reduce_retracing=True
    )
    def predict(data):
        logits, probs = predict_fn(state, data)
        return logits, probs
    module.predict = predict

    tf.saved_model.save(module, TF_MODEL_PATH,
        signatures={
            'predict': module.predict.get_concrete_function(),
        },
        options=tf.saved_model.SaveOptions(save_debug_info=False)
    )

def export_model_to_tf_lite():
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_PATH)
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
    coreml_pipeline = ct.PassPipeline()
    coreml_pipeline.remove_passes([ "common::merge_consecutive_reshapes" ])

    coreml_model = ct.convert(TF_MODEL_PATH,
        source='TensorFlow',
        # debug=True,
        # inputs=[ct.TensorType(dtype=np.float16)],
        # outputs=[ct.TensorType(dtype=np.float16), ct.TensorType(dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS17,
        pass_pipeline=coreml_pipeline)
    
    # Kind of hacky output renaming, but better here then in the app...
    spec = coreml_model.get_spec()
    ct.utils.rename_feature(spec, "Identity", "logits")
    ct.utils.rename_feature(spec, "Identity_1", "probs")
    coreml_model = ct.models.model.MLModel(spec, weights_dir=coreml_model.weights_dir)

    coreml_model.save("Audio2Midi.mlpackage")

if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')

    jax.config.update('jax_default_prng_impl', 'unsafe_rbg')

    current_directory = Path(__file__).resolve().parent
    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    model, state = load_newest_checkpoint(
        checkpoint_path,
        model_replication=False # Disable model sharding as it is not supported by coremlutils
    )

    print("Exporting the model itself...")
    export_model_to_tf(model, state)
    # export_model_to_tf_lite()
    export_model_to_coreml()
