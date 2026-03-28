"""Benchmark fp8_fixed_scale_pointwise_conv_call vs native bf16 on the actual model."""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

from model import OutputSequenceGenerator, model_config
from infer import change_fp_precision
from rope import precompute_frequencies, RopeFreqs
from fp8_ops import fp8_linear_call, fp8_fixed_scale_pointwise_conv_call

FORWARD_DTYPE = jnp.bfloat16

def make_model(key):
    return OutputSequenceGenerator(conf=model_config, key=key)

def benchmark_forward_backward(model, audio, rope_freqs, label, n_warmup=2, n_iter=5):
    model_bf16 = change_fp_precision(model, dtype=FORWARD_DTYPE)

    @eqx.filter_jit
    def fwd_bwd(model, audio, rope_freqs):
        @eqx.filter_grad
        def grad_fn(model):
            (logits, _probs), state = jax.vmap(
                model, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch"
            )(audio, None, rope_freqs)
            return jnp.mean(logits)
        return grad_fn(model)

    # Warmup (includes compilation)
    print(f"\n[{label}] Warming up ({n_warmup} iters, includes JIT compile)...", flush=True)
    for i in range(n_warmup):
        t0 = time.perf_counter()
        grads = fwd_bwd(model_bf16, audio, rope_freqs)
        jax.block_until_ready(grads)
        t1 = time.perf_counter()
        print(f"  warmup {i}: {t1-t0:.2f}s", flush=True)

    # Timed runs
    print(f"[{label}] Timing {n_iter} iterations...", flush=True)
    times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        grads = fwd_bwd(model_bf16, audio, rope_freqs)
        jax.block_until_ready(grads)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  iter {i}: {t1-t0:.3f}s", flush=True)

    avg = sum(times) / len(times)
    print(f"[{label}] Average: {avg:.3f}s per fwd+bwd", flush=True)
    return avg

def main():
    key = jax.random.key(42)
    model_key, data_key = jax.random.split(key)

    print("Creating model...", flush=True)
    model = make_model(model_key)

    sample_rate = 32000  # Must match AudioToMidiDatasetLoader.SAMPLE_RATE
    audio_duration = 5.0
    batch_size = 32
    audio = jax.random.normal(data_key, (batch_size, 2, int(sample_rate * audio_duration)), dtype=FORWARD_DTYPE)
    rope_freqs = precompute_frequencies(64, 250)

    # --- FP8 benchmark (current model.py uses fp8_linear_call) ---
    t_fp8 = benchmark_forward_backward(model, audio, rope_freqs, "FP8 Linear")

    # --- BF16 benchmark (patch all fp8 calls → direct calls) ---
    import fp8_ops
    import model as model_mod
    original_fp8_linear_call = fp8_ops.fp8_linear_call
    original_fp8_fixed_pw_conv_call = fp8_ops.fp8_fixed_scale_pointwise_conv_call
    def bypass_linear(linear, x):
        # eqx.nn.Linear only handles 1D; for 2D, use matmul directly
        out = x @ linear.weight.T
        if linear.bias is not None:
            out = out + linear.bias
        return out
    def bypass_pw_conv(conv, x):
        return conv(x)
    fp8_ops.fp8_linear_call = bypass_linear
    fp8_ops.fp8_fixed_scale_pointwise_conv_call = bypass_pw_conv
    model_mod.fp8_linear_call = bypass_linear
    model_mod.fp8_fixed_scale_pointwise_conv_call = bypass_pw_conv

    t_bf16 = benchmark_forward_backward(model, audio, rope_freqs, "BF16 Linear")

    # Restore
    fp8_ops.fp8_linear_call = original_fp8_linear_call
    fp8_ops.fp8_fixed_scale_pointwise_conv_call = original_fp8_fixed_pw_conv_call
    model_mod.fp8_linear_call = original_fp8_linear_call
    model_mod.fp8_fixed_scale_pointwise_conv_call = original_fp8_fixed_pw_conv_call

    print(f"\n{'='*50}")
    print(f"FP8 Linear:  {t_fp8:.3f}s per fwd+bwd")
    print(f"BF16 Linear: {t_bf16:.3f}s per fwd+bwd")
    if t_fp8 > 0:
        print(f"Speedup: {t_bf16/t_fp8:.2f}x (>1 means FP8 is faster)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
