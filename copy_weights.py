# Utility for copying weights from a pretrained model to a new model

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import equinox as eqx
import numpy as np

from infer import load_newest_checkpoint
from model import OutputSequenceGenerator
from audio_to_midi_dataset import get_data_prep_config

from pathlib import Path


# Load the existing model with the currently specified architecture
pretrained_model, pretrained_state = load_newest_checkpoint(
    "/Volumes/git/ml/audio-to-midi/audio_to_midi_checkpoints",
    ensemble_select=False,
)

new_config = {
    "dims": [18, 36, 72, 144, 288, 576, 1152],
    "depths": [3, 3, 3, 3, 3, 21, 3],

    "num_transformer_layers": 4,
    "num_transformer_heads": 4,
    "attention_size": 64,
    "compressed_attention_q_size": 64,
    "compressed_attention_kv_size": 64,
    "transformer_dropout_rate": 0.1,

    "sdd_rate": 0.1,
}
main_keys = jnp.stack([
    jax.random.key(1),
])
@eqx.filter_vmap(out_axes=(eqx.if_array(0), eqx.if_array(0)))
def make_ensemble(key):
    # init_key_1, init_key_2 = jax.random.split(key, num=2)
    model, state = eqx.nn.make_with_state(OutputSequenceGenerator)(new_config, key)
    # model = init_model(model, key=init_key_2)
    return model, state
new_model, new_state = make_ensemble(main_keys)

# This is a bit of a hack, but we simply flatten the pytree to its leaves, and
# replace any matching leaves in a greedy fashion with the pretrained model
new_model_leaves = jax.tree_util.tree_leaves(new_model)
updated_leaves = [
    pretrained_leaf if (
        hasattr(new_leaf, "shape") and hasattr(new_leaf, "dtype") and
        hasattr(pretrained_leaf, "shape") and hasattr(pretrained_leaf, "dtype") and
        new_leaf.shape == pretrained_leaf.shape and
        new_leaf.dtype == pretrained_leaf.dtype
    ) else new_leaf
    for new_leaf, pretrained_leaf
    in zip(new_model_leaves, jax.tree_util.tree_leaves(pretrained_model))
]
# We may need to pad the leaves due to the zip
updated_leaves = updated_leaves + new_model_leaves[len(updated_leaves):]

# Report some stats about how many leaves were updated
num_copied_leaves = 0
num_new_init_leaves = 0
for candidate, fresh_init_leaf in zip(updated_leaves, new_model_leaves):
    if np.all(candidate == fresh_init_leaf):
        num_new_init_leaves += 1
    else:
        num_copied_leaves += 1
print(f"#copied leaves: {num_copied_leaves}, #new: {num_new_init_leaves}")

# Rebuild the new model from the updated leaves
new_model = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(new_model), updated_leaves)

# Save the new model
current_directory = Path(__file__).resolve().parent
checkpoint_path = current_directory / "new_model_checkpoint"
checkpoint_options = ocp.CheckpointManagerOptions()
with ocp.CheckpointManager(
    checkpoint_path,
    options=checkpoint_options,
    item_names=('params', 'state'),
    metadata={
        'model': new_config,
        'data_prep': get_data_prep_config(),
    },
) as checkpoint_manager:
    checkpoint_manager.save(
        0,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(eqx.filter(new_model, eqx.is_inexact_array)),
            state=ocp.args.StandardSave(new_state),
        ),
    )
