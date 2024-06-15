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

from audio_to_midi_dataset import AudioToMidiDatasetLoader, visualize_sample, MODEL_AUDIO_LENGTH
from model import OutputSequenceGenerator, model_config, get_model_metadata
from infer import detailed_event_loss

@eqx.filter_jit
def compute_loss_from_output(logits, expected_output):
    # Prioritize the initial attacks a lot more in the loss than the roll-off
    only_attack_logits = jnp.select([expected_output > 0.95], [logits], -100.0)
    only_attack_expected = jnp.select([expected_output > 0.95], [expected_output], 0.0)
    attack_loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(only_attack_logits, only_attack_expected)

    full_loss = jax.vmap(optax.sigmoid_binary_cross_entropy)(logits, expected_output)

    return 2 * jnp.sum(attack_loss) + jnp.sum(full_loss)

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

@eqx.filter_jit
def compute_training_step(
    model, state, audio, expected_outputs, opt_state, key, tx,
):
    key, new_key = jax.random.split(key)
    (loss, state), grads = compute_loss(
        model,
        state,
        audio=audio,
        expected_outputs=expected_outputs,
        key=key,
    )

    updates, update_opt_state = tx.update(grads, opt_state, model)
    update_model = eqx.apply_updates(model, updates)

    return loss, update_model, state, update_opt_state, new_key

def compute_model_output_frames(model, state):
    # TODO(knielsen): Find a better way of doing this
    (find_shape_output_logits, _), _ = model(
        jnp.zeros((2, int(AudioToMidiDatasetLoader.SAMPLE_RATE * MODEL_AUDIO_LENGTH))),
        state
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

def compute_testset_loss_individual(model, state, testset_dir: Path, num_model_output_frames: int, key: jax.random.PRNGKey, sharding, batch_size=32):
    inference_model = eqx.nn.inference_mode(model)
    batches = load_test_set(testset_dir, num_model_output_frames, sharding, batch_size=batch_size)
    print("Loaded test set")

    loss_map = {}
    for sample_name, audio, midi_events in batches:
        test_loss_keys = jax.random.split(key, num=audio.shape[0])
        (logits, probs), _new_state = jax.vmap(inference_model, in_axes=(0, None, 0), out_axes=(0, None))(audio, state, test_loss_keys)
        test_losses = jax.vmap(compute_loss_from_output)(logits, midi_events)

        stitched_probs = np.concatenate(probs, axis=0)
        stitched_events = np.concatenate(midi_events, axis=0)

        detailed_loss = detailed_event_loss(stitched_probs, stitched_events)
        loss_map[sample_name] = {
            "loss": np.mean(test_losses),
            "hit_rate": detailed_loss.hit_rate,
            "eventized_diff": detailed_loss.full_diff,
            "phantom_note_diff": detailed_loss.phantom_notes_diff,
            "missed_note_diff": detailed_loss.missed_notes_diff,
        }

    print("Finished evaluating test loss")
    return loss_map

def compute_testset_loss(model, state, testset_dir: Path, num_model_output_frames, key: jax.random.PRNGKey, sharding, batch_size=32):
    per_sample_map = compute_testset_loss_individual(model, state, testset_dir, num_model_output_frames, key, sharding, batch_size)

    test_loss = jnp.zeros((1,))
    hit_rate = jnp.zeros((1,))
    eventized_diff = jnp.zeros((1,))
    count = jnp.array(0, dtype=jnp.int32)
    for losses in per_sample_map.values():
        test_loss += losses["loss"]
        hit_rate += losses["hit_rate"]
        eventized_diff += losses["eventized_diff"]
        count += 1

    return (test_loss / count)[0], (hit_rate / count)[0], (eventized_diff / count)[0]

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
    model,
    state,
    tx,
    data_loader,
    opt_state: optax.OptState,
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

    loss_sum = jnp.array([0.0], dtype=jnp.float32)
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
        recovery_model = model
        recovery_opt_state = opt_state

        loss, model, state, opt_state, key = compute_training_step(
            model,
            state,
            audio,
            events,
            opt_state,
            key,
            tx,
        )
        step_end_time = time.time()

        if jnp.isnan(loss):
            print(f"Encountered NAN loss at step {step}. Trying to recover!")
            model = recovery_model
            opt_state = recovery_opt_state
            continue

        if checkpoint_manager.should_save(step):
            filtered_model = eqx.filter(model, eqx.is_array)
            checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    params=ocp.args.StandardSave(filtered_model),
                    state=ocp.args.StandardSave(state),
                ),
                metrics=testset_hitrates,
            )

        loss_sum = loss_sum + loss

        if step % print_every == 0 and step != 0:
            learning_rate = learning_rate_schedule(step)

            averaged_loss = (loss_sum / print_every)[0]

            if trainloss_csv is not None:
                trainloss_csv.writerow([step, averaged_loss, step_end_time - start_time, step * audio.shape[0], learning_rate])
            print(f"Step {step}/{num_steps}, Loss: {averaged_loss}, LR = {learning_rate}")

            loss_sum = jnp.array([0.0], dtype=jnp.float32)

        if step % testset_loss_every == 0:
            print("Evaluating test losses...")
            for (name, testset_dir) in testset_dirs.items():
                eval_key, key = jax.random.split(key, num=2)
                testset_loss, hit_rate, eventized_diff = compute_testset_loss(model, state, testset_dir, num_model_output_frames, eval_key, batch_sharding)
                if testloss_csv is not None:
                    testloss_csv.writerow([name, step, testset_loss, step_end_time - start_time, step * audio.shape[0]])
                testset_hitrates[name] = float(hit_rate)
                print(f"Test loss {name}: {testset_loss}, hit_rate = {hit_rate}, eventized_diff = {eventized_diff}")

    return model, state, opt_state

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

def main():
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

    current_directory = Path(__file__).resolve().parent
    dataset_dir = Path("/Volumes/git/ml/datasets/midi-to-sound/true_harmonic")
    testset_dirs = {
        'validation_set': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set"),
        'validation_sets_only_yamaha': Path("/Volumes/git/ml/datasets/midi-to-sound/validation_set"),
    }

    num_devices = len(jax.devices())

    batch_size = 4 * num_devices
    num_steps = 1000000
    learning_rate_schedule = create_learning_rate_schedule(2.5 * 1e-4, 1000, num_steps)

    checkpoint_every = 10
    checkpoints_to_keep = 3
    dataset_num_workers = 2
    dataset_prefetch_count = 20

    key = jax.random.PRNGKey(1234)
    model_init_key, training_key, dataset_loader_key = jax.random.split(key, num=3)

    print(f"Running on {num_devices} devices with an effective batch size of {batch_size}")

    audio_to_midi, state = eqx.nn.make_with_state(OutputSequenceGenerator)(model_config, model_init_key)

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

    step_to_restore = checkpoint_manager.latest_step()
    if step_to_restore is not None:
        print(f"Restoring saved model at step {step_to_restore}")
        model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
        restored_map = checkpoint_manager.restore(
            step_to_restore,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(model_params),
                state=ocp.args.StandardRestore(state),
            ),
        )

        model_params = restored_map["params"]
        state = restored_map["state"]
        
        audio_to_midi = eqx.combine(model_params, static_model)

    # Replicate the model on all JAX devices
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh_replicate_everywhere = Mesh(device_mesh, axis_names=("_"))
    replicate_everywhere = NamedSharding(mesh_replicate_everywhere, PartitionSpec())

    model_params, static_model = eqx.partition(audio_to_midi, eqx.is_array)
    model_params = jax.device_put(model_params, replicate_everywhere)
    audio_to_midi = eqx.combine(model_params, static_model)

    tx = optax.adamw(learning_rate=learning_rate_schedule)
    tx = optax.chain(optax.clip_by_global_norm(5.0), tx)
    # The filtering is necessary to have the opt-state flattening working
    opt_state = tx.init(eqx.filter(audio_to_midi, eqx.is_inexact_array))

    num_model_output_frames = compute_model_output_frames(audio_to_midi, state)
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

            audio_to_midi, state, opt_state = train(
                audio_to_midi,
                state,
                tx,
                dataset_loader_iter,
                opt_state,
                checkpoint_manager,
                trainloss_csv=trainloss_csv,
                testloss_csv=testloss_csv,
                learning_rate_schedule=learning_rate_schedule,
                device_mesh=device_mesh,
                num_model_output_frames=num_model_output_frames, # TODO: Consider getting rid of this
                testset_dirs=testset_dirs,
                num_steps=num_steps,
                print_every=1,
                key=training_key,
                testset_loss_every=checkpoint_every,
            )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_threefry_partitionable", True)

    # with jax.profiler.trace("/tmp/jax-trace"):
    main()
