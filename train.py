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
from model import OutputSequenceGenerator, model_config, get_model_metadata
from infer import detailed_event_loss

from rope import precompute_frequencies

@eqx.filter_jit
def compute_loss_from_output(logits, expected_output):
    loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(logits, expected_output)
    return jnp.sum(loss)

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(model, state, cos_freq, sin_freq, audio, expected_outputs, key):
    batch_size = audio.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    (logits, probs), state = jax.vmap(
        model, in_axes=(0, None, None, None, 0, None), out_axes=(0, None), axis_name="batch",
    )(audio, state, cos_freq, sin_freq, batched_keys, True)

    loss = jax.vmap(compute_loss_from_output)(logits, expected_outputs)
    return jnp.mean(loss), state

@eqx.filter_jit
@eqx.filter_vmap(
    # TODO: Handle vmap'ed keys
    in_axes=(eqx.if_array(0), eqx.if_array(0), None, None, None, None, eqx.if_array(0), None, None),
    out_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None),
)
def compute_training_step(
    model_ensemble, state, cos_freq, sin_freq, audio, expected_outputs, opt_state, key, tx,
):
    key, new_key = jax.random.split(key)
    (loss, state), grads = compute_loss(
        model_ensemble,
        state,
        cos_freq=cos_freq,
        sin_freq=sin_freq,
        audio=audio,
        expected_outputs=expected_outputs,
        key=key,
    )

    updates, update_opt_state = tx.update(grads, opt_state, model_ensemble)
    update_model = eqx.apply_updates(model_ensemble, updates)

    return loss, update_model, state, update_opt_state, new_key

@eqx.filter_vmap(in_axes=(eqx.if_array(0), eqx.if_array(0), None, None))
def compute_model_output_frames(model, state, cos_freq: jax.Array, sin_freq: jax.Array):
    # TODO(knielsen): Find a better way of doing this
    (find_shape_output_logits, _), _ = model(
        samples=jnp.zeros((2, int(AudioToMidiDatasetLoader.SAMPLE_RATE * MODEL_AUDIO_LENGTH))),
        state=state,
        cos_freq=cos_freq,
        sin_freq=sin_freq,
    )
    num_model_output_frames = find_shape_output_logits.shape[0]
    return num_model_output_frames

@lru_cache(maxsize=1)
def load_test_set(testset_dir: Path, num_model_output_frames: int, sharding, batch_size: int):
    sample_names = AudioToMidiDatasetLoader.load_sample_names(testset_dir)

    batches = []
    for sample_name in sample_names:
        midi_events, audio, _sample_names = AudioToMidiDatasetLoader.load_samples(testset_dir, num_model_output_frames, [sample_name])
        batches.append((sample_name, audio, midi_events))
    return batches

def compute_testset_loss_individual(model_ensemble, state_ensemble, cos_freq, sin_freq, testset_dir: Path, num_model_output_frames: int, key: jax.random.PRNGKey, sharding, batch_size=32):
    batches = load_test_set(testset_dir, num_model_output_frames, sharding, batch_size=batch_size)
    print("Loaded test set")

    @eqx.filter_vmap(
        in_axes=(eqx.if_array(0), eqx.if_array(0), None, None, None, None),
        out_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0)),
    )
    def run_inference_single_model(model, state, cos_freq, sin_freq, audio, midi_events):
        inference_model = eqx.nn.inference_mode(model)
        (logits, probs), _new_state = jax.vmap(inference_model, in_axes=(0, None, None, None, 0), out_axes=(0, None))(audio, state, cos_freq, sin_freq, test_loss_keys)
        test_losses = jax.vmap(compute_loss_from_output)(logits, midi_events)
        return logits, probs, test_losses

    loss_map = {}
    for sample_name, audio, midi_events in batches:
        test_loss_keys = jax.random.split(key, num=audio.shape[0])
        logits_all, probs_all, test_losses_all = run_inference_single_model(model_ensemble, state_ensemble, cos_freq, sin_freq, audio, midi_events)

        test_losses = []
        hit_rates = []
        eventized_diffs = []
        phantom_note_diffs = []
        missed_note_diffs = []
        for _logits, probs, np_test_losses in zip(logits_all, probs_all, test_losses_all):
            stitched_probs = np.concatenate(probs, axis=0)
            stitched_events = np.concatenate(midi_events, axis=0)

            detailed_loss = detailed_event_loss(stitched_probs, stitched_events)
            test_losses.append(np.mean(np_test_losses))
            hit_rates.append(detailed_loss.hit_rate)
            eventized_diffs.append(detailed_loss.full_diff)
            phantom_note_diffs.append(detailed_loss.phantom_notes_diff)
            missed_note_diffs.append(detailed_loss.missed_notes_diff)

        loss_map[sample_name] = {
            "loss": np.array(test_losses),
            "hit_rate": np.array(hit_rates),
            "eventized_diff": np.array(eventized_diffs),
            "phantom_note_diff": np.array(phantom_note_diffs),
            "missed_note_diff": np.array(missed_note_diffs),
        }

    print("Finished evaluating test loss")
    return loss_map

def compute_testset_loss(model_ensemble, state_ensemble, cos_freq, sin_freq, testset_dir: Path, num_model_output_frames, key: jax.random.PRNGKey, sharding, batch_size=32):
    per_sample_map = compute_testset_loss_individual(model_ensemble, state_ensemble, cos_freq, sin_freq, testset_dir, num_model_output_frames, key, sharding, batch_size)

    test_loss = np.zeros_like(list(per_sample_map.values())[0]["loss"])
    hit_rate = np.zeros_like(list(per_sample_map.values())[0]["hit_rate"])
    eventized_diff = np.zeros_like(list(per_sample_map.values())[0]["eventized_diff"])

    count = 0
    for losses in per_sample_map.values():
        test_loss += losses["loss"]
        hit_rate += losses["hit_rate"]
        eventized_diff += losses["eventized_diff"]
        count += 1

    return (test_loss / count), (hit_rate / count), (eventized_diff / count)

@partial(jax.jit, donate_argnames=["samples"])
def add_sample_noise(
    samples, key: jax.random.PRNGKey
) -> Float[Array, "channels samples"]:
    """In order to make overfitting less likely this function perturbs the audio sampel in various ways:
    1. Add gausian noise
    """
    key1, key2 = jax.random.split(key, num=2)
    sigma = jax.random.uniform(key1) / 10  # Randomize the level of noise
    gaussian_noise = sigma * jax.random.normal(key2, samples.shape)
    
    return samples + gaussian_noise

def train(
    model_ensemble,
    state_ensemble,
    tx,
    cos_freq: jax.Array,
    sin_freq: jax.Array,
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
        else 0
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

    for step, batch in zip(range(start_step, num_steps + 1), data_loader):
        key, noise_key = jax.random.split(key, 2)

        (audio, events) = jax.device_put(
            (batch["audio"], batch["events"]),
            batch_sharding,
        )
        audio = add_sample_noise(audio, noise_key)

        # Keep the old model state in memory until we are sure the loss is not nan
        recovery_model = model_ensemble
        recovery_opt_state = opt_state_ensemble

        loss, model_ensemble, state_ensemble, opt_state_ensemble, key = compute_training_step(
            model_ensemble,
            state_ensemble,
            cos_freq,
            sin_freq,
            audio,
            events,
            opt_state_ensemble,
            key,
            tx,
        )
        step_end_time = time.time()

        if jnp.any(jnp.isnan(loss)):
            print(f"Encountered NAN loss at step {step}. Trying to recover!")
            model_ensemble = recovery_model
            opt_state_ensemble = recovery_opt_state
            continue

        if checkpoint_manager.should_save(step):
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

            loss_sum = make_loss_sum(model_ensemble) 

        if step % testset_loss_every == 0:
            print("Evaluating test losses...")
            testset_losses = []
            for (name, testset_dir) in testset_dirs.items():
                eval_key, key = jax.random.split(key, num=2)
                testset_loss, hit_rate, eventized_diff = compute_testset_loss(model_ensemble, state_ensemble, cos_freq, sin_freq, testset_dir, num_model_output_frames, eval_key, batch_sharding)
                if testloss_csv is not None:
                    testloss_csv.writerow([name, step, testset_loss, step_end_time - start_time, step * audio.shape[0]])
                # testset_hitrates[name] = float(hit_rate)
                print(f"Test loss {name}: {testset_loss}, hit_rate = {hit_rate}, eventized_diff = {eventized_diff}")
                testset_losses.append(testset_loss)

            # Recombine
            # TODO(knielsen): Refactor this! 
            # TODO: Consider sum of testset losses
            # TODO: Reset optimizer state?
            recombination_key, key = jax.random.split(key, num=2)
            model_ensemble = evolve_model_ensemble(model_ensemble, testset_losses[0], recombination_key)

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
    def mutate_leaf(leaf: jax.Array, index_to_mutate: int, key: jax.random.PRNGKey, mutation_rate = 0.005):
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
        recombination_rate = 0.00001  # 0,001% chance of recombining

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


def main():
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/home/knielsen/ml/datasets/midi-to-sound/varried")
    # dataset_dir = Path("/home/knielsen/ml/datasets/midi-to-sound/varried/true_melodic")
    testset_dirs = {
        'validation_set': Path("/home/knielsen/ml/datasets/validation_set"),
        'validation_sets_only_yamaha': Path("/home/knielsen/ml/datasets/validation_set_only_yamaha"),
    }

    num_devices = len(jax.devices())

    batch_size = 16 * num_devices
    num_steps = 100_000
    learning_rate_schedule = create_learning_rate_schedule(5 * 1e-4, 1000, num_steps)
    num_models = 6

    checkpoint_every = 1000
    checkpoints_to_keep = 3
    dataset_num_workers = 2
    dataset_prefetch_count = 20

    main_key = jax.random.PRNGKey(1234)
    model_init_key, training_key, dataset_loader_key, recombination_key = jax.random.split(main_key, num=4)

    print(f"Running on {num_devices} devices with an effective batch size of {batch_size}")

    @eqx.filter_vmap(out_axes=(eqx.if_array(0), eqx.if_array(0)))
    def make_ensemble(key):
        return eqx.nn.make_with_state(OutputSequenceGenerator)(model_config, key)

    ensemble_keys = jax.random.split(model_init_key, num_models)
    audio_to_midi_ensemble, model_states = make_ensemble(ensemble_keys)
    print(audio_to_midi_ensemble)

    checkpoint_path = current_directory / "audio_to_midi_checkpoints"
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpoints_to_keep,
        save_interval_steps=checkpoint_every,
        best_mode='max',
        best_fn=score_by_checkpoint_metrics,
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path,
        options=checkpoint_options,
        item_names=('params', 'state'),
        metadata=get_model_metadata()
    )

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    # mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    # replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())

    # model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
    # model_params = jax.device_put(model_params, replicate_everywhere)
    # audio_to_midi = eqx.combine(model_params, static_model)
    # state = jax.device_put(state, replicate_everywhere)

    tx = optax.adamw(learning_rate=learning_rate_schedule)
    tx = optax.chain(optax.clip_by_global_norm(5.0), tx)
    # The filtering is necessary to have the opt-state flattening working

    @eqx.filter_vmap
    def make_opt_states(model):
        return tx.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state_ensemble = make_opt_states(audio_to_midi_ensemble)

    cos_freq, sin_freq = precompute_frequencies(model_config["attention_size"], 200)  # TODO: Fix hardcoded number

    num_model_output_frames = compute_model_output_frames(audio_to_midi_ensemble, model_states, cos_freq, sin_freq)
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
                audio_to_midi_ensemble,
                model_states,
                tx,
                cos_freq,
                sin_freq,
                dataset_loader_iter,
                opt_state_ensemble,
                checkpoint_manager,
                trainloss_csv=trainloss_csv,
                testloss_csv=testloss_csv,
                learning_rate_schedule=learning_rate_schedule,
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
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace"):
    main()
