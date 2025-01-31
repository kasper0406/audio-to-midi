from pathlib import Path

import equinox as eqx
import jax
import equinox.internal as eqxi
import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

from infer import load_newest_checkpoint
import jax.numpy as jnp

import coremltools as ct
from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

# TODO: Figure out a way to not hardcode this
SAMPLE_RATE = 32_000
DURATION = 2.0


def export_model_to_coreml(model, state):
    context = jax_mlir.make_ir_context()
    input_sample_count = int(DURATION * SAMPLE_RATE)
    example_samples = jnp.zeros((2, input_sample_count))

    @jax.jit
    def infer_fn(samples):
        return eqxi.finalise_fn(model.predict)(state, samples)

    jax_exported = export(infer_fn)(example_samples)
    hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
    coreml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=DEFAULT_HLO_PIPELINE,
    )

    # Kind of hacky output renaming, but better here then in the app...
    # spec = coreml_model.get_spec()
    # ct.utils.rename_feature(spec, "Identity", "logits")
    # ct.utils.rename_feature(spec, "Identity_1", "probs")
    # coreml_model = ct.models.model.MLModel(spec, weights_dir=coreml_model.weights_dir)

    coreml_model.save("Audio2Midi.mlpackage")


if __name__ == "__main__":
    jax.config.update('jax_default_prng_impl', 'unsafe_rbg')

    current_directory = Path(__file__).resolve().parent
    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    model, state = load_newest_checkpoint(
        checkpoint_path,
        model_replication=False  # Disable model sharding as it is not supported by coremlutils
    )

    print("Exporting the model itself...")
    export_model_to_coreml(model, state)

    print("Done exporting model!")
