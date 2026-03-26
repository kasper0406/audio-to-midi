"""Per-tensor FP8 dynamic scaling for matmul and conv operations.

Uses custom_vjp to independently quantize each operation's inputs to fp8,
compute with f32 accumulation (preferred_element_type), and dequantize back.
Inter-layer gradients remain in bf16/f32, avoiding the cross-layer overflow
that makes naive fp8 training impossible in deep models.

Empirically validated: 0 NaN at depth=9 with w_scale=0.3 (matches bf16 stability).
"""
import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial

fp8_e4m3 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(fp8_e4m3).max)  # 448.0

# 1D conv dimension numbers: (batch, channel, spatial)
_DN_1D = lax.ConvDimensionNumbers(
    lhs_spec=(0, 1, 2), rhs_spec=(0, 1, 2), out_spec=(0, 1, 2)
)


def quantize_to_fp8(x):
    """Per-tensor current scaling: compute scale from max(|x|), quantize to fp8.

    Returns (x_fp8, scale) where x ≈ x_fp8 * scale.
    Scale is stop_gradient'd — it's a fixed constant, not differentiable.
    """
    amax = jax.lax.stop_gradient(jnp.max(jnp.abs(x.astype(jnp.float32))))
    scale = jnp.maximum(amax, 1e-12) / FP8_MAX
    x_scaled = x.astype(jnp.float32) / scale
    x_fp8 = jnp.clip(x_scaled, -FP8_MAX, FP8_MAX).astype(fp8_e4m3)
    return x_fp8, scale


def _fp8_dot_dequant(a_fp8, b_fp8, a_scale, b_scale):
    """FP8 dot with f32 accumulate, dequantized."""
    out = jnp.dot(a_fp8, b_fp8, preferred_element_type=jnp.float32)
    return out * (a_scale * b_scale)


@jax.custom_vjp
def fp8_matmul(x, w):
    """FP8 matrix multiply with per-tensor dynamic scaling.

    Forward:  quantize(x), quantize(w) → fp8 dot + f32 accum → dequantize
    Backward: each backward matmul independently quantizes its inputs.

    Args:
        x: activation tensor, any float dtype (typically bf16)
        w: weight matrix, any float dtype (typically bf16)
    Returns:
        result in x.dtype
    """
    x_fp8, x_s = quantize_to_fp8(x)
    w_fp8, w_s = quantize_to_fp8(w)
    return _fp8_dot_dequant(x_fp8, w_fp8, x_s, w_s).astype(x.dtype)


def _fp8_matmul_fwd(x, w):
    result = fp8_matmul(x, w)
    return result, (x, w)


def _fp8_matmul_bwd(res, g):
    x, w = res

    # grad_x = g @ w.T — quantize g and w independently
    g_fp8, g_s = quantize_to_fp8(g)
    w_fp8, w_s = quantize_to_fp8(w)
    grad_x = jnp.dot(g_fp8, w_fp8.T, preferred_element_type=jnp.float32)
    grad_x = (grad_x * (g_s * w_s)).astype(x.dtype)

    # grad_w = x.T @ g — quantize x and g independently
    x_fp8, x_s = quantize_to_fp8(x)
    # Handle both unbatched (1D/2D from vmap) and batched cases
    if x.ndim == 1:
        # x: (K,), g: (N,) → grad_w: (K, N) via outer product
        grad_w = jnp.outer(x_fp8.astype(jnp.float32), g_fp8.astype(jnp.float32))
    elif x.ndim == 2:
        grad_w = jnp.dot(x_fp8.T, g_fp8, preferred_element_type=jnp.float32)
    else:
        # Batched: x: (B, T, K), g: (B, T, N) → grad_w: (K, N)
        grad_w = jnp.einsum('btk,btn->kn', x_fp8.astype(jnp.float32), g_fp8.astype(jnp.float32))
    grad_w = (grad_w * (x_s * g_s)).astype(w.dtype)

    return grad_x, grad_w


fp8_matmul.defvjp(_fp8_matmul_fwd, _fp8_matmul_bwd)


def make_fp8_conv1d(padding='SAME', stride=1, groups=1):
    """Create an fp8 conv1d function with closed-over non-JAX args.

    The returned function takes (x, weight, bias_or_none) and computes
    conv1d in fp8 with per-tensor scaling.

    Forward:  fp8 conv with f32 accumulate
    Backward grad_x: fp8 conv_transpose with f32 accumulate
    Backward grad_w: bf16 vjp (weight grad less perf-critical)
    """

    @jax.custom_vjp
    def fp8_conv(x, w):
        x_fp8, x_s = quantize_to_fp8(x)
        w_fp8, w_s = quantize_to_fp8(w)
        out = lax.conv_general_dilated(
            x_fp8, w_fp8, (stride,), padding,
            dimension_numbers=_DN_1D, feature_group_count=groups,
            preferred_element_type=jnp.float32,
        )
        return (out * (x_s * w_s)).astype(x.dtype)

    def fwd(x, w):
        return fp8_conv(x, w), (x, w)

    def bwd(res, g):
        x, w = res

        # For grouped convs or strided convs, fall back to bf16 vjp for grad_x
        # (conv_transpose doesn't support feature_group_count, and strided
        # conv_transpose can produce shape mismatches)
        if groups > 1 or stride > 1:
            def conv_fwd_bf16(x_, w_):
                return lax.conv_general_dilated(
                    x_, w_, (stride,), padding,
                    dimension_numbers=_DN_1D, feature_group_count=groups,
                )
            _, vjp_fn = jax.vjp(conv_fwd_bf16, x.astype(jnp.bfloat16), w.astype(jnp.bfloat16))
            grad_x, grad_w = vjp_fn(g.astype(jnp.bfloat16))
            return grad_x.astype(x.dtype), grad_w.astype(w.dtype)

        # grad_x: conv_transpose with fp8 (stride=1, groups=1 only)
        g_fp8, g_s = quantize_to_fp8(g)
        w_fp8, w_s = quantize_to_fp8(w)
        grad_x_f32 = lax.conv_transpose(
            g_fp8, w_fp8, (stride,), padding,
            dimension_numbers=_DN_1D,
            transpose_kernel=True,
            preferred_element_type=jnp.float32,
        )
        grad_x = (grad_x_f32 * (g_s * w_s)).astype(x.dtype)

        # grad_w: bf16 vjp (simpler, correct for all configs)
        def conv_fwd_for_w(w_):
            return lax.conv_general_dilated(
                x.astype(jnp.bfloat16), w_, (stride,), padding,
                dimension_numbers=_DN_1D, feature_group_count=groups,
            )
        grad_w = jax.vjp(conv_fwd_for_w, w.astype(jnp.bfloat16))[1](g.astype(jnp.bfloat16))[0]
        grad_w = grad_w.astype(w.dtype)

        return grad_x, grad_w

    fp8_conv.defvjp(fwd, bwd)
    return fp8_conv


def fp8_linear_call(linear_module, x):
    """Call an eqx.nn.Linear using fp8_matmul, preserving bias.

    eqx.nn.Linear stores weight as (out_features, in_features).
    """
    out = fp8_matmul(x, linear_module.weight.T)
    if linear_module.use_bias:
        out = out + linear_module.bias
    return out


def fp8_conv1d_call(conv_module, x):
    """Call an eqx.nn.Conv1d using fp8 conv, preserving bias.

    Reads padding/stride/groups from the conv module and builds the
    appropriate fp8_conv function.

    eqx.nn.Conv1d works on unbatched (C, L) tensors. We add/remove the
    batch dimension for lax.conv_general_dilated which requires 3D input.

    eqx.nn.Conv1d stores:
      - weight: (out_channels, in_channels/groups, kernel_size)
      - bias: (out_channels, 1) or None
      - padding: 'SAME' or ((pad_lo, pad_hi),)
      - stride: (stride,)
      - groups: int
    """
    padding = conv_module.padding
    stride = conv_module.stride[0]
    groups = conv_module.groups

    fp8_conv = make_fp8_conv1d(padding=padding, stride=stride, groups=groups)

    # Add batch dim: (C, L) → (1, C, L)
    x_batched = x[None, :, :]
    out = fp8_conv(x_batched, conv_module.weight)
    # Remove batch dim: (1, C', L') → (C', L')
    out = out[0]

    if conv_module.use_bias:
        out = out + conv_module.bias  # bias is (out_channels, 1), broadcasts over spatial
    return out
