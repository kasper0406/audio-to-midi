from functools import partial, lru_cache
from pathlib import Path
from typing import Optional, Callable
import os
import time
import sys
from typing import Dict
import copy

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
from dataclasses import dataclass

from audio_to_midi_dataset import AudioToMidiDatasetLoader, visualize_sample, MODEL_AUDIO_LENGTH
from model import OutputSequenceGenerator, model_config, get_model_metadata, SelfAttention
from infer import detailed_event_loss, change_fp_precision
from grain_loader import TransformSettings, AudioToMidiSource, create_dataset_loader
from rope import precompute_frequencies, RopeFreqs

from metrics import configure_tensorboard
from tensorboardX import SummaryWriter

MODEL_DTYPE = jnp.float32
FORWARD_DTYPE = jnp.float16
BACKWARD_DTYPE = jnp.float16

@eqx.filter_jit(donate="all")
def compute_loss_from_output(logits, expected_output, scale):
    # loss = jax.vmap(partial(optax.losses.poly_loss_cross_entropy, epsilon=-1.0))(logits, expected_output)
    # jax.debug.print("logits = {l}", l=logits)
    loss = jax.vmap(optax.losses.sigmoid_binary_cross_entropy)(logits, expected_output)
    scaled_loss = loss * scale
    # jax.debug.print("loss = {l}, scaled_loss = {sl}", l=loss, sl=scaled_loss)

    return jnp.sum(scaled_loss)

@eqx.filter_jit(donate="all-except-first")
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(model_back_precision, state, audio, rope_freqs, expected_outputs, scale, key):
    batch_size = audio.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    model_back_precision = change_fp_precision(model_back_precision, dtype=FORWARD_DTYPE)
    audio = audio.astype(dtype=FORWARD_DTYPE)
    (logits, probs), state = jax.vmap(
        model_back_precision, in_axes=(0, None, None, 0, None), out_axes=(0, None), axis_name="batch",
    )(audio, state, rope_freqs, batched_keys, True)

    logits = logits.astype(dtype=jnp.float32)  # Calculate the actual losses with float32 precision
    loss = jax.vmap(compute_loss_from_output, in_axes=(0, 0, None))(logits, expected_outputs, scale)
    return jnp.mean(loss), state

@eqx.filter_vmap(in_axes=(None, None, None, eqx.if_array(0), eqx.if_array(0), None))
def compute_model_output_frames(batch_size, sample_rate: int, audio_duration: float, model, state, rope_freqs: RopeFreqs):
    # TODO(knielsen): Find a better way of doing this
    (find_shape_output_logits, _), _ = jax.vmap(model, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch")(
        jnp.zeros((batch_size, 2, int(sample_rate * audio_duration))),
        state,
        rope_freqs,
    )
    num_model_output_frames = find_shape_output_logits.shape[1]  # Output 0 is the batch size
    return num_model_output_frames

@lru_cache
def load_test_set(testset_dir: Path, num_model_output_frames: int, sharding, batch_size: int):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)

    batches = []
    for sample_name in sample_names:
        # print(f"Loading {sample_name}")
        midi_events, audio, _sample_names = AudioToMidiDatasetLoader.load_samples(testset_dir, num_model_output_frames, [sample_name], AudioToMidiDatasetLoader.SAMPLE_RATE, MODEL_AUDIO_LENGTH, skip_cache=True)
        batches.append((sample_name, audio, midi_events))
    return batches

def compute_testset_loss_individual(
    model_ensemble,
    state_ensemble,
    rope_freqs: RopeFreqs,
    testset_dir: Path,
    num_model_output_frames: int,
    key: jax.random.PRNGKey,
    sharding,
    batch_size: int = 32,
):
    batches = load_test_set(testset_dir, num_model_output_frames, sharding, batch_size=batch_size)
    print("Loaded test set")

    @eqx.filter_jit(donate="all-except-first")
    def testset_loss_function(logits, expected_output):
        loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(logits, expected_output)
        return jnp.sum(loss)

    # @eqx.filter_jit(donate="all-except-first")
    @eqx.filter_jit
    @eqx.filter_vmap(
        in_axes=(eqx.if_array(0), eqx.if_array(0), None, None),
        out_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0))
    )
    def run_inference_single_model(inference_model, state, audio, midi_events):
        # Compute the full batched model, even though we only compute one audio sample
        pretend_batch_audio = jnp.zeros((1, *audio.shape))
        pretend_batch_audio = pretend_batch_audio.at[0, ...].set(audio)

        pretend_batch_midi_events = jnp.zeros((1, *midi_events.shape))
        pretend_batch_midi_events = pretend_batch_midi_events.at[0, ...].set(midi_events)

        (logits, probs), _new_state = jax.vmap(inference_model, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch")(pretend_batch_audio, state, rope_freqs)
        test_losses = jax.vmap(testset_loss_function)(logits, pretend_batch_midi_events)
        return logits[0, ...], probs[0, ...], test_losses[0, ...]

    generate_visualizations = len(batches) < 30

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

            detailed_loss = detailed_event_loss(stitched_probs, stitched_events, generate_visualization=generate_visualizations)
            test_losses.append(np.mean(np_test_losses))
            hit_rates.append(detailed_loss.hit_rate)
            eventized_diffs.append(detailed_loss.full_diff)
            phantom_note_diffs.append(detailed_loss.phantom_notes_diff)
            missed_note_diffs.append(detailed_loss.missed_notes_diff)

            if generate_visualizations:
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

def compute_testset_loss(
    model_ensemble,
    state_ensemble,
    rope_freqs: RopeFreqs,
    testset_dir: Path,
    num_model_output_frames,
    key: jax.random.PRNGKey,
    sharding,
    batch_size: int = 32,
):
    per_sample_map = compute_testset_loss_individual(
        model_ensemble,
        state_ensemble,
        rope_freqs,
        testset_dir,
        num_model_output_frames,
        key,
        sharding,
        batch_size,
    )

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
    learning_rate_schedule: Callable,
    device_mesh: [],
    num_model_output_frames: int,
    rope_freqs: RopeFreqs,
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

    @eqx.filter_jit(donate="all-except-first")
    @eqx.filter_vmap(
        # TODO: Handle vmap'ed keys
        in_axes=(None, eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None, None, None, None),
        out_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None, eqx.if_array(0), eqx.if_array(0)),
    )
    def compute_training_step(
        tx: optax.GradientTransformation,
        flat_model, flat_state, flat_opt_state, audio, expected_outputs, key, grad_scale,
    ):
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        state = jax.tree_util.tree_unflatten(treedef_state, flat_state)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

        key, new_key = jax.random.split(key)
        model_backward_dtype = change_fp_precision(model, dtype=BACKWARD_DTYPE)
        (scaled_loss, update_state), scaled_grads = compute_loss(
            model_backward_dtype,
            state,
            audio=audio,
            rope_freqs=jax.tree_util.tree_map(lambda x: x, rope_freqs),
            expected_outputs=expected_outputs,
            scale=jnp.array(grad_scale, dtype=jnp.float16),
            key=key,
        )

        # max_grad = jax.tree_util.tree_reduce(lambda acc, x: jnp.maximum(acc, jnp.max(jnp.abs(x))), scaled_grads, initializer=0.0)
        # jax.debug.print("L1 scaled grads: {l1}", l1=max_grad)

        scaled_grads = change_fp_precision(scaled_grads, dtype=MODEL_DTYPE)
        grads = jax.tree_util.tree_map(lambda g: g / grad_scale, scaled_grads)
        grads_valid = jnp.all(jnp.array(
            [jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]
        ))

        updates, update_opt_state = tx.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        update_model = eqx.apply_updates(model, updates)

        update_flat_model = jax.tree_util.tree_leaves(update_model)
        update_flat_opt_state = jax.tree_util.tree_leaves(update_opt_state)
        update_update_state = jax.tree_util.tree_leaves(update_state)

        loss = scaled_loss / grad_scale
        return loss, update_flat_model, update_update_state, update_flat_opt_state, new_key, grads_valid, scaled_loss

    def copy_pytree(tree):
        def copy_leaf(x):
            if isinstance(x, jnp.ndarray):
                return np.copy(x) # Copy in main memory
            return copy.deepcopy(x)
        return jax.tree.map(copy_leaf, tree)

    recovery_model = copy_pytree(flat_model)
    recovery_state = copy_pytree(flat_state)
    recovery_opt_state = copy_pytree(flat_opt_state)
    grad_scale = 1.0
    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        key, noise_key = jax.random.split(key, 2)

        # print("Putting data on GPU...")
        (events, audio) = jax.device_put(batch, batch_sharding)

        # Keep the old model state in memory until we are sure the loss is not nan
        # make sure to copy them so we can do donation
        if step % 100 == 0:
            recovery_model = copy_pytree(flat_model)
            recovery_state = copy_pytree(flat_state)
            recovery_opt_state = copy_pytree(flat_opt_state)

        # print(f"Executing step {step}")
        loss, flat_model, flat_state, flat_opt_state, key, grads_valid, scaled_loss = compute_training_step(
            tx,
            flat_model, 
            flat_state,
            flat_opt_state,
            audio,
            events,
            key,
            jnp.array(grad_scale, dtype=jnp.float16),
        )
        # print(f"Finished executing step {step}")

        if not np.all(grads_valid) or not np.all(np.isfinite(loss)):
            new_grad_scale = grad_scale / 2
            print(f"Encountered NAN/inf loss at step {step}, loss = {loss}. Trying to recover! Gradscale {grad_scale} -> {new_grad_scale}")

            grad_scale = new_grad_scale
            flat_model = recovery_model
            flat_state = recovery_state
            flat_opt_state = recovery_opt_state
            continue

        if (scaled_loss < 10_000).all(): # TODO: Make this configurable
            new_grad_scale = grad_scale * 2
            print(f"Grad scale: {grad_scale} -> {new_grad_scale}")
            grad_scale = new_grad_scale

        if checkpoint_manager.should_save(step):
            model_ensemble = jax.tree.unflatten(treedef_model, flat_model)
            state_ensemble = jax.tree.unflatten(treedef_state, flat_state)
            filtered_model = eqx.filter(model_ensemble, eqx.is_inexact_array)
            checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    params=ocp.args.StandardSave(filtered_model),
                    state=ocp.args.StandardSave(state_ensemble),
                ),
            )

        loss_sum = loss_sum + loss

        if step % print_every == 0 and step != 0:
            learning_rate = learning_rate_schedule(step)

            averaged_loss = (loss_sum / print_every)[0]

            print(f"Step {step}/{num_steps}, Loss: {averaged_loss}, LR = {learning_rate}")

            # Pick the average loss of the best model in the ensemble
            summary_writer.add_scalar("train/loss", jnp.min(averaged_loss), step)
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
                testset_loss, hit_rate, eventized_diff, visualizations = compute_testset_loss(
                    model_ensemble,
                    state_ensemble,
                    rope_freqs,
                    testset_dir,
                    num_model_output_frames,
                    eval_key,
                    batch_sharding,
                )
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
            recombination_key, key = jax.random.split(key, num=2)
            ensemble_testset_losses = np.mean(np.stack(testset_losses), axis=0)
            print(f"Ensemble scores: {ensemble_testset_losses}")
            model_ensemble = evolve_model_ensemble(model_ensemble, ensemble_testset_losses, recombination_key)

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
        boundaries=[warmup_steps],
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
    head_weight_std = 0.2
    cnn_weight_std = 0.2
    cnn_bias_std = 0.01

    def flatten(lst):
        return [item for sublist in lst for item in sublist]
    
    head_weight_key, head_bias_key, cnn_weight_key, cnn_bias_key = jax.random.split(key, num=4)

    def extract_field(objs, field: str):
        res = []
        for obj in objs:
            if hasattr(obj, field):
                value = getattr(obj, field)
                if value is not None:
                    res.append(value)
        return res

    # Initialize head weights
    def is_self_attention(node):
        return isinstance(node, SelfAttention)
    get_head_weights = lambda m: flatten([
        extract_field([x.query_down_proj, x.query_up_proj, x.kv_down_proj, x.key_up_proj, x.value_up_proj], "weight")
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_self_attention) if is_self_attention(x)
    ])
    head_weights = get_head_weights(model)
    new_head_weights = [
        jax.random.normal(subkey, weight.shape) * head_weight_std
        for weight, subkey in zip(head_weights, jax.random.split(head_weight_key, len(head_weights)))
    ]
    model = eqx.tree_at(get_head_weights, model, new_head_weights)

    # Initialize head biases
    get_head_biases = lambda m: flatten([
        extract_field([x.query_down_proj, x.query_up_proj, x.kv_down_proj, x.key_up_proj, x.value_up_proj], "bias")
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_self_attention) if is_self_attention(x)
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
    cnn_biases = get_cnn_biases(model)
    new_cnn_biases = [
        jax.random.normal(subkey, weight.shape) * cnn_bias_std
        for weight, subkey in zip(cnn_biases, jax.random.split(cnn_bias_key, len(cnn_biases)))
    ]
    model = eqx.tree_at(get_cnn_biases, model, new_cnn_biases)

    return model

def setup_optimizers(model, base_learning_rate: float, layer_lr_decay: float, weight_decay: float, warmup_steps: int, num_steps: int):
    # Implement layer learning-rate decay by figuring out the depth from the PyTree path and adjusting the optimizer to the depth
    def depth_extracting_label_fn(tree):
        def map_fn(path, value):
            if path[0].name == "layers":
                # We are inside our Conv-blocks
                conv_depth = path[1].idx
                layer_depth = path[3].idx

                # Compute the prefix sum, and add the current layer depth
                computed_depth = 0
                for i in range(conv_depth):
                    computed_depth += model_config["depths"][i]
                computed_depth += layer_depth

                # print(f"Conv path {path} at depth {computed_depth}")
                return f"conv_layer|{computed_depth}"
            elif path[0].name == "transformer":
                seq_nr = path[2].idx
                # We are inside the transformer stack
                # print(f"Transformer path {path} at depth {seq_nr}")
                return f"transformer_layer|{seq_nr}"
            else:
                return f"default|0"

        return jax.tree_util.tree_map_with_path(map_fn, tree)

    def max_depth(acc, value: str):
        # print(f"Value: {value}, acc: {acc}")
        layername, depth = value.split("|")
        depth = int(depth)

        if layername == "conv_layer":
            return max(acc[0], depth), acc[1]
        elif layername == "transformer_layer":
            return acc[0], max(acc[1], depth)
        else:
            return acc
    max_conv_depth, max_transformer_depth = jax.tree_util.tree_reduce(
        max_depth,
        depth_extracting_label_fn(model), (0, 0),
    )
    print(f"Max conv depth: {max_conv_depth}, max transformer depth: {max_transformer_depth}")

    conv_lr_by_depth = { f"conv_layer|{depth}": create_learning_rate_schedule(base_learning_rate * (layer_lr_decay ** (max_conv_depth - depth)), warmup_steps, num_steps) for depth in range(max_conv_depth + 1) }
    # transformer_lr_by_depth = { f"transformer_layer|{depth}": create_learning_rate_schedule(base_learning_rate * (layer_lr_decay ** (max_transformer_depth - depth)), warmup_steps, num_steps) for depth in range(max_transformer_depth + 1) }
    transformer_lr_by_depth = { f"transformer_layer|{depth}": create_learning_rate_schedule(base_learning_rate, warmup_steps, num_steps) for depth in range(max_transformer_depth + 1) }
    default_lr = { f"default|0": create_learning_rate_schedule(base_learning_rate, warmup_steps, num_steps) for depth in range(max_transformer_depth) }
    learning_rates_by_depth = conv_lr_by_depth | transformer_lr_by_depth | default_lr
    tx = optax.multi_transform({
        depth: optax.adamw(lr_schedule, weight_decay=weight_decay, eps=1e-3, b1=0.9, b2=0.999)
        for depth, lr_schedule in learning_rates_by_depth.items()
    }, depth_extracting_label_fn)
    tx = optax.chain(tx, optax.clip_by_global_norm(1.0))

    return tx, default_lr["default|0"]


def main():
    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/test/")
    testset_dirs = {
        'validation_set': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set"),
        'validation_sets_only_yamaha': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set_only_yamaha"),
        'validation_set_generated': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set_generated"),
    }

    num_devices = len(jax.devices())

    batch_size = 8 * num_devices
    num_steps = 200_000
    warmup_steps = 1000
    base_learning_rate = 1 * 1e-4
    layer_lr_decay = 0.7
    weight_decay = 0.005
    model_init_keys = jnp.stack([
        jax.random.key(1),
        # jax.random.key(2),
        # jax.random.key(3),
        # jax.random.key(4),
        # jax.random.key(5),
    ])

    transform_settings = TransformSettings(
        pan_probability=0.8,
        channel_switch_probability=0.5,
        cut_probability=0.4,
        rotate_probability=0.9,
        random_erasing_probability=0.3,
        mixup_probability=0.6,
        gain_probability=0.8,
        noise_probability=0.8,
        label_smoothing_alpha=0.005,
    )

    checkpoint_every = 20
    checkpoints_to_keep = 3
    dataset_num_workers = 3

    main_key = jax.random.PRNGKey(1234)
    training_key, = jax.random.split(main_key, num=1)

    print(f"Running on {num_devices} devices with an effective batch size of {batch_size}")
    
    summary_writer = configure_tensorboard()
    h_params = model_config
    h_params["train/batch_size"] = batch_size
    h_params["train/total_steps"] = num_steps
    h_params["train/warmup_steps"] = warmup_steps
    summary_writer.add_hparams(h_params, {})

    rope_freqs = precompute_frequencies(model_config["attention_size"], 300)

    @eqx.filter_vmap(out_axes=(eqx.if_array(0), eqx.if_array(0)))
    def make_ensemble(key):
        init_key_1, init_key_2 = jax.random.split(key, num=2)
        model, state = eqx.nn.make_with_state(OutputSequenceGenerator)(model_config, init_key_1)
        # model = init_model(model, key=init_key_2)
        return model, state

    audio_to_midi_ensemble, model_states = make_ensemble(model_init_keys)
    audio_to_midi_ensemble = change_fp_precision(audio_to_midi_ensemble, dtype=MODEL_DTYPE)
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

    print("Setting up optimizers...")
    tx, base_learning_rate_schedule = setup_optimizers(audio_to_midi_ensemble, base_learning_rate, layer_lr_decay, weight_decay, warmup_steps, num_steps)

    @eqx.filter_vmap
    def make_opt_states(model):
        # The filtering is necessary to have the opt-state flattening working
        return tx.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state_ensemble = make_opt_states(audio_to_midi_ensemble)

    sample_rate = AudioToMidiDatasetLoader.SAMPLE_RATE
    audio_duration = MODEL_AUDIO_LENGTH
    num_model_output_frames = compute_model_output_frames(batch_size, sample_rate, audio_duration, audio_to_midi_ensemble, model_states, rope_freqs)
    print(f"Model output frames: {num_model_output_frames}")

    print("Setting up dataset loader...")
    dataset_loader = create_dataset_loader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=dataset_num_workers,
        num_epochs=100000,
        sample_rate=sample_rate,
        duration=audio_duration,
        output_divisions=num_model_output_frames,
        transform_settings=transform_settings,
    )
    dataset_loader_iter = iter(dataset_loader)

    print("Starting training...")
    audio_to_midi_ensemble, model_states, opt_state = train(
        summary_writer,
        audio_to_midi_ensemble,
        model_states,
        tx,
        dataset_loader_iter,
        opt_state_ensemble,
        checkpoint_manager,
        learning_rate_schedule=base_learning_rate_schedule,
        device_mesh=device_mesh,
        rope_freqs=rope_freqs,
        num_model_output_frames=num_model_output_frames, # TODO: Consider getting rid of this
        testset_dirs=testset_dirs,
        num_steps=num_steps,
        print_every=25,
        key=training_key,
        testset_loss_every=checkpoint_every,
    )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9'
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_gemm=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        '--xla_gpu_all_reduce_combine_threshold_bytes=51200 '
        '--xla_gpu_graph_level=0 '
    )

    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    jax.threefry_partitionable(True)
    jax.default_matmul_precision("BF16_BF16_BF16")

    # with jax.profiler.trace("/tmp/jax-trace"):
    main()
