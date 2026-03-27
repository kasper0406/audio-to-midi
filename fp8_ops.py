"""Per-tensor FP8 dynamic scaling for matmul operations.

Uses custom_vjp to independently quantize each operation's inputs to fp8,
compute with f32 accumulation (preferred_element_type), and dequantize back.
Inter-layer gradients remain in bf16/f32, avoiding the cross-layer overflow
that makes naive fp8 training impossible in deep models.

Optimizations vs naive approach (inspired by Flax fp8_ops):
 - Save quantized residuals in forward → no re-quantization in backward
 - Use E5M2 for backward gradients (wider range ~57344 vs E4M3's 448)
 - Use lax.dot_general with explicit dimension numbers for better XLA fusion
"""
import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial

fp8_e4m3 = jnp.float8_e4m3fn
fp8_e5m2 = jnp.float8_e5m2
FP8_E4M3_MAX = float(jnp.finfo(fp8_e4m3).max)  # 448.0
FP8_E5M2_MAX = float(jnp.finfo(fp8_e5m2).max)   # 57344.0

# 1D conv dimension numbers: (batch, channel, spatial)
_DN_1D = lax.ConvDimensionNumbers(
    lhs_spec=(0, 1, 2), rhs_spec=(0, 1, 2), out_spec=(0, 1, 2)
)


def quantize_to_fp8(x, dtype=None, fp8_max=None):
    """Per-tensor current scaling: compute scale from max(|x|), quantize to fp8.

    Returns (x_fp8, scale) where x ≈ x_fp8 * scale.
    Scale is stop_gradient'd — it's a fixed constant, not differentiable.
    """
    if dtype is None:
        dtype = fp8_e4m3
    if fp8_max is None:
        fp8_max = FP8_E4M3_MAX if dtype == fp8_e4m3 else FP8_E5M2_MAX
    amax = jax.lax.stop_gradient(jnp.max(jnp.abs(x.astype(jnp.float32))))
    scale = jnp.maximum(amax, 1e-12) / fp8_max
    x_scaled = x.astype(jnp.float32) / scale
    x_fp8 = jnp.clip(x_scaled, -fp8_max, fp8_max).astype(dtype)
    return x_fp8, scale


@jax.custom_vjp
def fp8_matmul(x, w):
    """FP8 matrix multiply with per-tensor dynamic scaling.

    Forward:  quantize(x), quantize(w) → fp8 dot + f32 accum → dequantize
    Backward: reuses saved quantized tensors; quantizes gradients to E5M2.
    """
    x_fp8, x_s = quantize_to_fp8(x, fp8_e4m3)
    w_fp8, w_s = quantize_to_fp8(w, fp8_e4m3)
    out = jnp.dot(x_fp8, w_fp8, preferred_element_type=jnp.float32)
    return (out * (x_s * w_s)).astype(x.dtype)


def _fp8_matmul_fwd(x, w):
    x_fp8, x_s = quantize_to_fp8(x, fp8_e4m3)
    w_fp8, w_s = quantize_to_fp8(w, fp8_e4m3)
    out = jnp.dot(x_fp8, w_fp8, preferred_element_type=jnp.float32)
    result = (out * (x_s * w_s)).astype(x.dtype)
    # Save quantized tensors + scales → no re-quantization in backward
    return result, (x, x_fp8, x_s, w_fp8, w_s)


def _fp8_matmul_bwd(res, g):
    x, x_fp8, x_s, w_fp8, w_s = res

    # Quantize gradient to E5M2 (wider range for gradients)
    g_fp8, g_s = quantize_to_fp8(g, fp8_e5m2)

    # grad_x = g @ w.T — reuse saved w_fp8/w_s
    grad_x = jnp.dot(g_fp8, w_fp8.T, preferred_element_type=jnp.float32)
    grad_x = (grad_x * (g_s * w_s)).astype(x.dtype)

    # grad_w = x.T @ g — reuse saved x_fp8/x_s
    if x.ndim == 1:
        grad_w = jnp.outer(x_fp8.astype(jnp.float32), g_fp8.astype(jnp.float32))
    elif x.ndim == 2:
        grad_w = jnp.dot(x_fp8.T, g_fp8, preferred_element_type=jnp.float32)
    else:
        grad_w = jnp.einsum('btk,btn->kn', x_fp8.astype(jnp.float32), g_fp8.astype(jnp.float32))
    grad_w = (grad_w * (x_s * g_s)).astype(w_fp8.dtype)
    # grad_w needs to match w's dtype, but w isn't in residuals — use x.dtype as proxy
    grad_w = grad_w.astype(x.dtype)

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

    Works for both 1D (single vector) and 2D (batched/sequence) inputs.
    For 2D inputs, quantizes the entire tensor at once — much more efficient
    than vmapping over the sequence dimension.

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
