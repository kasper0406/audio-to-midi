from pathlib import Path

import equinox as eqx
import jax
import equinox.internal as eqxi
import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export
import matplotlib.pyplot as plt

from infer import load_newest_checkpoint
from model import model_config
from rope import precompute_frequencies

import jax.numpy as jnp
import numpy as np

import coremltools as ct
from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

# TODO: Figure out a way to not hardcode this
SAMPLE_RATE = 16_000
DURATION = 5.0


def export_model_to_coreml(model, state):
    context = jax_mlir.make_ir_context()
    input_sample_count = int(DURATION * SAMPLE_RATE)
    example_samples = jnp.zeros((2, input_sample_count))

    rope_freqs = precompute_frequencies(model_config["attention_size"], 300)

    @jax.jit
    def infer_fn(samples):
        return eqxi.finalise_fn(model.predict)(state, samples, rope_freqs)

    jax_exported = export(infer_fn)(example_samples)
    hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)

    pass_pipeline = DEFAULT_HLO_PIPELINE
    pass_pipeline.remove_passes(["common::add_fp16_cast"])  # There are precision issues when fp16 casting intermediate calculations
    pass_pipeline.remove_passes(["common::const_elimination"])  # Sadly const_elimination currently makes the model fail to run!

    # pass_pipeline = ct.PassPipeline.EMPTY
    # pass_pipeline.append_pass("common::sanitize_input_output_names")
    # pass_pipeline.append_pass("common::const_elimination")

    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
    coreml_model = ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pass_pipeline,
    )
    print(coreml_model._mil_program)

    # print("Testing before rename")
    # prediction = coreml_model.predict({
    #     "_arg0": np.zeros((2, int(DURATION * SAMPLE_RATE))),
    # })
    # print("Done testing!")

    # Rename the input and output fields
    spec = coreml_model.get_spec()
    for input_description, new_name in zip(coreml_model.input_description, ["samples"]):
        ct.utils.rename_feature(spec, input_description, new_name)
    for ouput_description, new_name in zip(coreml_model.output_description, ["logits", "probs"]):
        ct.utils.rename_feature(spec, ouput_description, new_name)
    coreml_model = ct.models.model.MLModel(spec, weights_dir=coreml_model.weights_dir)

    coreml_model.save("Audio2Midi.mlpackage")

    return coreml_model


def plot_output(output):
    fig, ax1 = plt.subplots()

    X = jnp.arange(output.shape[0])
    Y = jnp.arange(output.shape[1])
    c = ax1.pcolor(X, Y, jnp.transpose(output))
    ax1.set(
        title=f"Output",
        xlabel="Temporal",
        ylabel="Features",
    )
    fig.colorbar(c)

    return fig


if __name__ == "__main__":
    jax.config.update('jax_default_prng_impl', 'unsafe_rbg')

    current_directory = Path(__file__).resolve().parent
    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    model, state = load_newest_checkpoint(
        checkpoint_path,
        model_replication=False  # Disable model sharding as it is not supported by coremlutils
    )
    
    test_samples = np.zeros((2, int(DURATION * SAMPLE_RATE)))
    rope_freqs = precompute_frequencies(model_config["attention_size"], 300)
    logits, probs = model.predict(state, test_samples, rope_freqs)
    plot_output(probs)

    print("Exporting the model itself...")
    cml_model = export_model_to_coreml(model, state)

    print("Done exporting model!")

    print("Attempting to call model...")

    print(f"New input description: {cml_model.input_description}")
    prediction = cml_model.predict({
        "samples": test_samples,
    })
    plot_output(prediction["probs"])
    plt.show(block = True)
