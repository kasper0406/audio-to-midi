from functools import partial, lru_cache
from pathlib import Path
from typing import Optional, Callable
import os
import csv
import time
import sys
from typing import Dict

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.flatten_util
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float
from functools import reduce
from more_itertools import chunked
import numpy as np
from collections import defaultdict

from audio_to_midi_dataset import AudioToMidiDatasetLoader, visualize_sample, MODEL_AUDIO_LENGTH
from resnext_model import OutputSequenceGenerator, model_config, get_model_metadata
from infer import detailed_event_loss

from rope import precompute_frequencies
from metrics import configure_tensorboard
from tensorboardX import SummaryWriter

@eqx.filter_jit
def compute_loss_from_output(logits, expected_output):
    loss = jax.vmap(partial(optax.sigmoid_focal_loss, alpha=None, gamma=1.0))(logits, expected_output)
    return jnp.sum(loss)

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(model, state, audio, expected_outputs, key):
    batch_size = audio.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    (logits, probs), state = jax.vmap(
        model, in_axes=(0, None, 0, None), out_axes=(0, None), axis_name="batch",
    )(audio, state, batched_keys, True)

    loss = jax.vmap(compute_loss_from_output)(logits, expected_outputs)
    return jnp.mean(loss), state

@eqx.filter_vmap(in_axes=(None, eqx.if_array(0), eqx.if_array(0)))
def compute_model_output_frames(batch_size, model, state):
    # TODO(knielsen): Find a better way of doing this
    (find_shape_output_logits, _), _ = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
        jnp.zeros((batch_size, 2, int(AudioToMidiDatasetLoader.SAMPLE_RATE * MODEL_AUDIO_LENGTH))),
        state,
    )
    num_model_output_frames = find_shape_output_logits.shape[1]  # Output 0 is the batch size
    return num_model_output_frames

@lru_cache
def load_test_set(testset_dir: Path, num_model_output_frames: int, sharding, batch_size: int):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)

    batches = []
    for sample_name in sample_names:
        midi_events, audio, _sample_names = AudioToMidiDatasetLoader.load_samples(testset_dir, num_model_output_frames, [sample_name], skip_cache=True)
        batches.append((sample_name, audio, midi_events))
    return batches

def compute_testset_loss_individual(model_ensemble, state_ensemble, testset_dir: Path, num_model_output_frames: int, key: jax.random.PRNGKey, sharding, batch_size=32):
    batches = load_test_set(testset_dir, num_model_output_frames, sharding, batch_size=batch_size)
    print("Loaded test set")

    @eqx.filter_jit
    def testset_loss_function(logits, expected_output):
        loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(logits, expected_output)
        return jnp.sum(loss)

    @eqx.filter_jit
    @eqx.filter_vmap(
        in_axes=(eqx.if_array(0), eqx.if_array(0), None, None, None, None),
        out_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0))
    )
    def run_inference_single_model(inference_model, state, audio, midi_events):
        # Compute the full batched model, even though we only compute one audio sample
        pretend_batch_audio = jnp.zeros((batch_size, *audio.shape))
        pretend_batch_audio = pretend_batch_audio.at[0, ...].set(audio)

        pretend_batch_midi_events = jnp.zeros((batch_size, *midi_events.shape))
        pretend_batch_midi_events = pretend_batch_midi_events.at[0, ...].set(midi_events)

        (logits, probs), _new_state = jax.vmap(inference_model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(pretend_batch_audio, state)
        test_losses = jax.vmap(testset_loss_function)(logits, pretend_batch_midi_events)
        return logits[0, ...], probs[0, ...], test_losses[0, ...]

    inference_model = eqx.nn.inference_mode(model_ensemble)
    loss_map = {}
    for sample_name, audios, midi_events in batches:
        logits_all = []
        probs_all = []
        test_losses_all = []
        for audio, midi_event in zip(audios, midi_events):
            logits, probs, test_losses = run_inference_single_model(inference_model, state_ensemble, audio, midi_event)

            if len(logits_all) == 0:
                # TODO: Nicer way to initialize?
                logits_all = [ [] for _i in range(logits.shape[0])]
                probs_all = [ [] for _i in range(logits.shape[0])]
                test_losses_all = [ [] for _i in range(logits.shape[0])]

            for i in range(logits.shape[0]):
                logits_all[i].append(logits[i, ...])
                probs_all[i].append(probs[i, ...])
                test_losses_all[i].append(test_losses[i, ...])

        test_losses = []
        hit_rates = []
        eventized_diffs = []
        phantom_note_diffs = []
        missed_note_diffs = []
        visualizations = []
        for _logits, probs, np_test_losses in zip(logits_all, probs_all, test_losses_all):
            stitched_probs = np.concatenate(probs, axis=0)
            stitched_events = np.concatenate(midi_events, axis=0)

            detailed_loss = detailed_event_loss(stitched_probs, stitched_events, generate_visualization=True)
            test_losses.append(np.mean(np_test_losses))
            hit_rates.append(detailed_loss.hit_rate)
            eventized_diffs.append(detailed_loss.full_diff)
            phantom_note_diffs.append(detailed_loss.phantom_notes_diff)
            missed_note_diffs.append(detailed_loss.missed_notes_diff)
            visualizations.append(detailed_loss.visualization)

        loss_map[sample_name] = {
            "loss": np.array(test_losses),
            "hit_rate": np.array(hit_rates),
            "eventized_diff": np.array(eventized_diffs),
            "phantom_note_diff": np.array(phantom_note_diffs),
            "missed_note_diff": np.array(missed_note_diffs),
            "visualizations": visualizations,
        }

    print("Finished evaluating test loss")
    return loss_map

def compute_testset_loss(model_ensemble, state_ensemble, testset_dir: Path, num_model_output_frames, key: jax.random.PRNGKey, sharding, batch_size=32):
    per_sample_map = compute_testset_loss_individual(model_ensemble, state_ensemble, testset_dir, num_model_output_frames, key, sharding, batch_size)

    test_loss = np.zeros_like(list(per_sample_map.values())[0]["loss"])
    hit_rate = np.zeros_like(list(per_sample_map.values())[0]["hit_rate"])
    eventized_diff = np.zeros_like(list(per_sample_map.values())[0]["eventized_diff"])
    visualizations = []

    count = 0
    for losses in per_sample_map.values():
        test_loss += losses["loss"]
        hit_rate += losses["hit_rate"]
        eventized_diff += losses["eventized_diff"]
        visualizations += losses["visualizations"]
        count += 1

    return (test_loss / count), (hit_rate / count), (eventized_diff / count), visualizations

def train(
    summary_writer: SummaryWriter,
    model_ensemble,
    state_ensemble,
    tx: optax.GradientTransformation,
    data_loader,
    opt_state_ensemble: optax.OptState,
    checkpoint_manager: ocp.CheckpointManager,
    trainloss_csv: Optional[any],
    testloss_csv: Optional[any],
    learning_rate_schedule: Callable,
    device_mesh: [],
    num_model_output_frames: int,
    testset_dirs: Dict[str, Path],
    num_steps: int = 10000,
    print_every: int = 1000,
    testset_loss_every: int = 1000,
    key: Optional[jax.random.PRNGKey] = None,
):
    start_time = time.time()

    start_step = (
        checkpoint_manager.latest_step() + 1
        if checkpoint_manager.latest_step() is not None
        else 1
    )

    batch_mesh = Mesh(device_mesh, ("batch",))
    batch_sharding = NamedSharding(
        batch_mesh,
        PartitionSpec(
            "batch",
        ),
    )

    @eqx.filter_vmap
    def make_loss_sum(models):
        return jnp.array([0.0], dtype=jnp.float32)
    loss_sum = make_loss_sum(model_ensemble)

    testset_hitrates = {}
    for name in testset_dirs.keys():
        testset_hitrates[name] = sys.float_info.max

    flat_model, treedef_model = jax.tree_util.tree_flatten(model_ensemble)
    flat_state, treedef_state = jax.tree_util.tree_flatten(state_ensemble)
    flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state_ensemble)

    @eqx.filter_jit
    @eqx.filter_vmap(
        # TODO: Handle vmap'ed keys
        in_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None, None, None, None),
        out_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None),
    )
    def compute_training_step(
        flat_model, flat_state, flat_opt_state, audio, expected_outputs, key, tx: optax.GradientTransformation,
    ):
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        state = jax.tree_util.tree_unflatten(treedef_state, flat_state)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

        key, new_key = jax.random.split(key)
        (loss, update_state), grads = compute_loss(
            model,
            state,
            audio=audio,
            expected_outputs=expected_outputs,
            key=key,
        )

        updates, update_opt_state = tx.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        update_model = eqx.apply_updates(model, updates)

        flat_update_model = jax.tree_util.tree_leaves(update_model)
        flat_update_state = jax.tree_util.tree_leaves(update_state)
        flat_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)

        return loss, flat_update_model, flat_update_state, flat_update_opt_state, new_key

    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        key, noise_key = jax.random.split(key, 2)

        (audio, events) = jax.device_put(
            (batch["audio"], batch["events"]),
            batch_sharding,
        )

        # Keep the old model state in memory until we are sure the loss is not nan
        recovery_model = flat_model
        recovery_state = flat_state
        recovery_opt_state = flat_opt_state

        loss, flat_model, flat_state, flat_opt_state, key = compute_training_step(
            flat_model, 
            flat_state,
            flat_opt_state,
            audio,
            events,
            key,
            tx,
        )
        step_end_time = time.time()

        if jnp.any(jnp.isnan(loss)):
            print(f"Encountered NAN loss at step {step}. Trying to recover!")
            flat_model = recovery_model
            flat_state = recovery_state
            flat_opt_state = recovery_opt_state
            continue

        if checkpoint_manager.should_save(step):
            model_ensemble = jax.tree.unflatten(treedef_model, flat_model)
            state_ensemble = jax.tree.unflatten(treedef_state, flat_state)
            filtered_model = eqx.filter(model_ensemble, eqx.is_array)
            checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    params=ocp.args.StandardSave(filtered_model),
                    state=ocp.args.StandardSave(state_ensemble),
                ),
                # metrics=testset_hitrates,
            )

        loss_sum = loss_sum + loss

        if step % print_every == 0 and step != 0:
            learning_rate = learning_rate_schedule(step)

            averaged_loss = (loss_sum / print_every)[0]

            if trainloss_csv is not None:
                trainloss_csv.writerow([step, averaged_loss, step_end_time - start_time, step * audio.shape[0], learning_rate])
            print(f"Step {step}/{num_steps}, Loss: {averaged_loss}, LR = {learning_rate}")
            
            summary_writer.add_scalar("train/loss", averaged_loss[0], step)
            summary_writer.add_scalar("train/learning_rate", learning_rate, step)
            summary_writer.flush()

            model_ensemble = jax.tree.unflatten(treedef_model, flat_model)
            loss_sum = make_loss_sum(model_ensemble) 

        if step % testset_loss_every == 0:
            model_ensemble = jax.tree.unflatten(treedef_model, flat_model)
            state_ensemble = jax.tree.unflatten(treedef_state, flat_state)

            print("Evaluating test losses...")
            testset_losses = []
            for (name, testset_dir) in testset_dirs.items():
                eval_key, key = jax.random.split(key, num=2)
                testset_loss, hit_rate, eventized_diff, visualizations = compute_testset_loss(model_ensemble, state_ensemble, testset_dir, num_model_output_frames, eval_key, batch_sharding)
                if testloss_csv is not None:
                    testloss_csv.writerow([name, step, testset_loss, step_end_time - start_time, step * audio.shape[0]])
                # testset_hitrates[name] = float(hit_rate)
                print(f"Test loss {name}: {testset_loss}, hit_rate = {hit_rate}, eventized_diff = {eventized_diff}")
                testset_losses.append(testset_loss)

                summary_writer.add_scalar(f"train/test-loss-{name}", testset_loss[0], step)
                for i, visualization in enumerate(visualizations):
                    summary_writer.add_figure(f"train/test-loss-{name}-{i}", visualization, step)
            summary_writer.flush()

            # Recombine
            # TODO(knielsen): Refactor this! 
            # TODO: Consider sum of testset losses
            # TODO: Reset optimizer state?
            # TODO(knielsen): Consider re-enabling recombination?
            # recombination_key, key = jax.random.split(key, num=2)
            # model_ensemble = evolve_model_ensemble(model_ensemble, testset_losses[0], recombination_key)

    model_ensemble = jax.tree.unflatten(treedef_model, flat_model)
    state_ensemble = jax.tree.unflatten(treedef_state, flat_state)
    opt_state_ensemble = jax.tree.unflatten(treedef_opt_state, flat_opt_state)

    return model_ensemble, state_ensemble, opt_state_ensemble

def create_learning_rate_schedule(base_learning_rate: float, warmup_steps: int, cosine_decay_steps: int):
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=base_learning_rate,
        transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_decay_steps
    )
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )

def score_by_checkpoint_metrics(metrics):
    mean_score = float(np.mean(np.array(list(metrics.values()))))
    return mean_score

def evolve_model_ensemble(model_ensemble, ensemble_scores, key: jax.random.PRNGKey):
    """
    Genetic algorithm to re-combine models into new models
    """
    def mutate_leaf(leaf: jax.Array, index_to_mutate: int, key: jax.random.PRNGKey, mutation_rate = 0.0005):
        if not eqx.is_array(leaf) or leaf.dtype not in (jnp.float16, jnp.float32):
            # Do not modify non-numpy arrays
            return leaf

        weights_to_mutate = leaf[index_to_mutate, ...]

        mutation_probs_key, normal_weights_key = jax.random.split(key, 2)
        mutation_probs = jax.random.uniform(mutation_probs_key, weights_to_mutate.shape)
        normal_weights = jax.random.normal(normal_weights_key, weights_to_mutate.shape, dtype=leaf.dtype)
        updated_weights = jax.lax.select(mutation_probs < mutation_rate, normal_weights, weights_to_mutate)

        return leaf.at[index_to_mutate, ...].set(updated_weights)

    def recombine(model_ensemble, parent_a_idx: int, parent_b_idx: int, result_idx: int, key: jax.random.PRNGKey):
        recombination_rate = 0.000001  # 0,0001% chance of recombining

        recombination_steps = 0
        current_parent_idx = 1  # Always start with parent_a weights (inversed in first recombination_steps sampling)

        def recombine_leaf(leaf, *rest):
            if not eqx.is_array(leaf) or leaf.dtype not in (jnp.float16, jnp.float32):
                # Do not modify non-numpy arrays
                return leaf

            # print(f"Should modify leaf with shape: {leaf.shape}")

            # Flatten the weights for easy copying
            parent_a_weights = leaf[parent_a_idx, ...].flatten()
            parent_b_weights = leaf[parent_b_idx, ...].flatten()

            nonlocal key  # We abuse the key from the outer function as we can not pass it along easily
            nonlocal recombination_steps
            nonlocal current_parent_idx

            counter = 0
            recombined_weights = jnp.zeros_like(parent_a_weights).flatten()
            while counter < parent_a_weights.shape[0]:
                if recombination_steps <= 0:
                    key, recombination_key = jax.random.split(key, 2)
                    recombination_steps = int(jax.random.geometric(recombination_key, recombination_rate, shape=tuple()))
                    current_parent_idx = (current_parent_idx + 1) % 2
                    print(f"Recombining after {recombination_steps} steps")

                # Figure out which parent to copy from
                current_parent = parent_a_weights
                if current_parent_idx != 0:
                    current_parent = parent_b_weights

                # Copy the weights in the recombination slice
                copy_end_idx = min(counter + recombination_steps, parent_a_weights.shape[0])
                indexes = slice(counter, copy_end_idx)
                recombined_weights = recombined_weights.at[indexes].set(current_parent[indexes])

                num_copied_elements = indexes.stop - indexes.start
                recombination_steps -= num_copied_elements
                counter += num_copied_elements

            # Update the leaf with the recombined weights
            recombined_weights = jnp.reshape(recombined_weights, leaf[result_idx, ...].shape)
            recombined_leaf = leaf.at[result_idx, ...].set(recombined_weights)

            key, mutation_key = jax.random.split(key, 2)
            return mutate_leaf(recombined_leaf, result_idx, mutation_key)

        return jax.tree.map(recombine_leaf, model_ensemble)

    if ensemble_scores.shape[0] <= 2:
        print("Not recombining due to low population")
        return model_ensemble

    recombined_ensemble = model_ensemble

    # print(f"Ensemble scores: {ensemble_scores}")
    # print(f"Sorted: {np.argsort(ensemble_scores)}")
    sorted_indices = list(np.argsort(ensemble_scores))
    # Keep the half best scoring models, and kill off the bottom half by recombination
    winner_indices = sorted_indices[0:(len(sorted_indices) // 2)]
    result_indices = sorted_indices[(len(sorted_indices) // 2):]
    for result_idx in result_indices:
        key, winner_key, recombine_key = jax.random.split(key, 3)

        random_integers = jax.random.randint(winner_key, shape=(100,), minval=0, maxval=len(winner_indices))
        parent_a_idx = winner_indices[int(random_integers[0])]
        # TODO(knielsen): Make this nicer...
        #                 Pick the first random winner that is not the same as parent_a
        i = 1
        while int(random_integers[0]) == int(random_integers[i]):
            i += 1
        parent_b_idx = int(winner_indices[random_integers[i]])
        print(f"Recombining {parent_a_idx} + {parent_b_idx} -> {result_idx}")

        recombined_ensemble = recombine(recombined_ensemble, parent_a_idx=parent_a_idx, parent_b_idx=parent_b_idx, result_idx=result_idx, key=recombine_key)

    return recombined_ensemble


def init_model(model, key: jax.random.PRNGKey):
    head_weight_std = 0.001
    cnn_weight_std = 0.02

    def flatten(lst):
        return [item for sublist in lst for item in sublist]
    
    head_weight_key, head_bias_key, cnn_weight_key, cnn_bias_key = jax.random.split(key, num=4)

    # Initialize head weights
    def is_multihead_attention(node):
        return isinstance(node, eqx.nn.MultiheadAttention)
    get_head_weights = lambda m: flatten([
        [x.query_proj.weight, x.key_proj.weight, x.value_proj.weight, x.output_proj.weight]
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_multihead_attention) if is_multihead_attention(x)
    ])
    head_weights = get_head_weights(model)
    new_head_weights = [
        jax.random.normal(subkey, weight.shape) * head_weight_std
        for weight, subkey in zip(head_weights, jax.random.split(head_weight_key, len(head_weights)))
    ]
    model = eqx.tree_at(get_head_weights, model, new_head_weights)

    # Initialize head biases
    get_head_biases = lambda m: flatten([
        [x.query_proj.bias, x.key_proj.bias, x.value_proj.bias, x.output_proj.bias]
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_multihead_attention) if is_multihead_attention(x)
    ])
    head_biases = get_head_biases(model)
    new_head_biases = [
        jnp.zeros_like(weight)
        for weight, subkey in zip(head_weights, jax.random.split(head_bias_key, len(head_biases)))
    ]
    model = eqx.tree_at(get_head_biases, model, new_head_biases)

    # Initialize CNN weights
    def is_conv_1d(node):
        return isinstance(node, eqx.nn.Conv1d)
    get_cnn_weights = lambda m: flatten([
        [x.weight]
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv_1d) if is_conv_1d(x)
    ])
    cnn_weights  = get_cnn_weights(model)
    new_cnn_weights = [
        jax.random.normal(subkey, weight.shape) * cnn_weight_std
        for weight, subkey in zip(cnn_weights, jax.random.split(cnn_weight_key, len(cnn_weights)))
    ]
    model = eqx.tree_at(get_cnn_weights, model, new_cnn_weights)

    # Initialize CNN biases
    get_cnn_biases = lambda m: flatten([
        [x.bias]
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv_1d) if is_conv_1d(x)
    ])
    cnn_biases  = get_cnn_biases(model)
    new_cnn_biases = [
        jnp.zeros_like(weight)
        for weight, subkey in zip(cnn_weights, jax.random.split(cnn_bias_key, len(cnn_biases)))
    ]
    model = eqx.tree_at(get_cnn_biases, model, new_cnn_biases)

    return model


def main():
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/test/")
    testset_dirs = {
        'validation_set': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set"),
        'validation_sets_only_yamaha': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set_only_yamaha"),
    }

    num_devices = len(jax.devices())

    batch_size = 4 * num_devices
    num_steps = 10000
    warmup_steps = 1000
    base_learning_rate = 5 * 1e-5
    layer_lr_decay = 0.9
    weight_decay = 1e-8
    num_models = 1

    checkpoint_every = 1000
    checkpoints_to_keep = 3
    dataset_num_workers = 2
    dataset_prefetch_count = 20

    main_key = jax.random.PRNGKey(1234)
    model_init_key, model_init_key_2, training_key, dataset_loader_key, recombination_key = jax.random.split(main_key, num=5)

    print(f"Running on {num_devices} devices with an effective batch size of {batch_size}")
    
    summary_writer = configure_tensorboard()
    h_params = model_config
    h_params["train/batch_size"] = batch_size
    h_params["train/total_steps"] = num_steps
    h_params["train/warmup_steps"] = warmup_steps
    summary_writer.add_hparams(h_params, {})

    @eqx.filter_vmap(out_axes=(eqx.if_array(0), eqx.if_array(0)))
    def make_ensemble(key):
        return eqx.nn.make_with_state(OutputSequenceGenerator)(model_config, key)

    ensemble_keys = jax.random.split(model_init_key, num_models)
    audio_to_midi_ensemble, model_states = make_ensemble(ensemble_keys)
    init_model(audio_to_midi_ensemble, key=model_init_key_2)
    print(audio_to_midi_ensemble)

    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpoints_to_keep,
        save_interval_steps=checkpoint_every,
        best_mode='max',
        # best_fn=score_by_checkpoint_metrics,
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path,
        options=checkpoint_options,
        item_names=('params', 'state'),
        metadata=get_model_metadata(),
    )

    # Load latest model
    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is not None:
        current_metadata = get_model_metadata()
        if current_metadata != checkpoint_manager.metadata():
            print(f"WARNING: The loaded model has metadata {checkpoint_manager.metadata()}")
            print(f"Current configuration is {current_metadata}")

        print(f"Restoring saved model at step {step_to_restore}")
        filtered_model, static_model = eqx.partition(audio_to_midi_ensemble, eqx.is_array)
        restored_map = checkpoint_manager.restore(
            step_to_restore,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(filtered_model),
                state=ocp.args.StandardRestore(model_states),
            ),
        )
        audio_to_midi_ensemble = eqx.combine(restored_map["params"], static_model)
        model_states = restored_map["state"]

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())

    # TODO(knielsen): Refactor to a function?
    model_params, static_model = eqx.partition(audio_to_midi_ensemble, eqx.is_array)
    model_params = jax.device_put(model_params, replicate_everywhere)
    audio_to_midi_ensemble = eqx.combine(model_params, static_model)
    model_states = jax.device_put(model_states, replicate_everywhere)

    # Implement layer learning-rate decay by figuring out the depth from the PyTree path and adjusting the optimizer to the depth
    def depth_extracting_label_fn(tree):
        def map_fn(path, value):
            first_seq = None
            for part in path:
                if isinstance(part, jax.tree_util.SequenceKey):
                    first_seq = part
                    break
            
            if first_seq:
                # print(f"Path {path} at depth {first_seq.idx}")
                return first_seq.idx
            return 0  # Default is depth 0 with standard learning rate

        return jax.tree_util.tree_map_with_path(map_fn, tree)

    learning_rates_by_depth = { depth: create_learning_rate_schedule(base_learning_rate * (layer_lr_decay ** depth), warmup_steps, num_steps) for depth in range(0, 10) }
    tx = optax.multi_transform({
        depth: optax.adamw(lr_schedule, weight_decay=weight_decay) for depth, lr_schedule in learning_rates_by_depth.items()
    }, depth_extracting_label_fn)
    
    # tx = optax.chain(optax.clip_by_global_norm(3.0), tx)

    @eqx.filter_vmap
    def make_opt_states(model):
        # The filtering is necessary to have the opt-state flattening working
        return tx.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state_ensemble = make_opt_states(audio_to_midi_ensemble)

    num_model_output_frames = compute_model_output_frames(batch_size, audio_to_midi_ensemble, model_states)
    print(f"Model output frames: {num_model_output_frames}")

    print("Setting up dataset loader...")
    dataset_loader = AudioToMidiDatasetLoader(
        num_model_output_frames=num_model_output_frames,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        prefetch_count=dataset_prefetch_count,
        key=dataset_loader_key,
        num_workers=dataset_num_workers,
        epochs=100000,
    )
    dataset_loader_iter = iter(dataset_loader)

    print("Starting training...")
    with open('train_loss.csv', mode='w', buffering=1) as trainloss_file:
        with open('test_loss.csv', mode='w', buffering=1) as testloss_file:
            trainloss_csv = csv.writer(trainloss_file)
            testloss_csv = csv.writer(testloss_file)

            audio_to_midi_ensemble, model_states, opt_state = train(
                summary_writer,
                audio_to_midi_ensemble,
                model_states,
                tx,
                dataset_loader_iter,
                opt_state_ensemble,
                checkpoint_manager,
                trainloss_csv=trainloss_csv,
                testloss_csv=testloss_csv,
                learning_rate_schedule=learning_rates_by_depth[0],
                device_mesh=device_mesh,
                num_model_output_frames=num_model_output_frames, # TODO: Consider getting rid of this
                testset_dirs=testset_dirs,
                num_steps=num_steps,
                print_every=25,
                key=training_key,
                testset_loss_every=checkpoint_every,
            )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace"):
    main()
