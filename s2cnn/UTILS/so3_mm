# @title
# so3_mm.py
# Updated for Python 3.12 / PyTorch 2.10
#
# Changes vs original:
#   - Removed all custom CUDA C kernel templates  -> torch.einsum on complex tensors
#   - Removed _cuda_SO3_mm autograd class         -> autograd works through einsum natively
#   - Removed s2cnn.utils.cuda dependency         -> no longer needed
#   - Removed s2cnn.utils.complex.complex_mm      -> torch.view_as_complex + .conj()
#   - Single code path for CPU and GPU            -> PyTorch dispatches to CUDA automatically
#
# Key difference from s2_mm.py:
#   s2_mm:  S2  -> SO(3)  output nspec LARGER  than input  (lifts a dimension)
#   so3_mm: SO(3) -> SO(3) output nspec SAME AS input       (stays in SO(3))
#
# Operation (derived from original CUDA kernel, conj_y=True, trans_y_spec=True):
#   For each degree l, contracting over spectral index p and feature_in k:
#
#     z[m, n, batch, f_out] = sum_{p, k}  x[m, p, batch, k]  *  conj(y[n, p, k, f_out])
#
#   In einsum: "mpbi, npif -> mnbf"  with y.conj()
#
# Where both x and y are viewed as [L, L, ...] per degree, with:
#   x: first L = output m, second L = contracted p
#   y: first L = output n, second L = contracted p

import math
import torch


def so3_mm(x, y):
    """
    SO(3) spectral matrix multiply: applies the kernel to the signal
    at each harmonic degree l, contracting over the shared spectral index p.

    :param x:      [l*m*n,   batch,      feature_in,  2]   signal  spectral coefficients
    :param y:      [l*m*n,   feature_in, feature_out, 2]   kernel  spectral coefficients
    :return:       [l*m*n,   batch,      feature_out, 2]   output  spectral coefficients
    """
    assert x.size(-1) == 2,          "last dim of x must be 2 (re, im)"
    assert y.size(-1) == 2,          "last dim of y must be 2 (re, im)"
    assert x.size(0)  == y.size(0),  f"nspec mismatch: x={x.size(0)}, y={y.size(0)}"
    assert x.size(2)  == y.size(1),  f"feature_in mismatch: x={x.size(2)}, y={y.size(1)}"

    nspec        = x.size(0)
    nbatch       = x.size(1)
    nfeature_in  = x.size(2)
    nfeature_out = y.size(2)

    # Recover nl from nspec = nl * (4*nl^2 - 1) / 3
    nl = round((3 / 4 * nspec) ** (1 / 3))
    assert nspec == nl * (4 * nl ** 2 - 1) // 3, \
        f"nspec={nspec} is not a valid SO(3) spectral size for any integer nl"

    # Pre-convert to complex once (avoids repeated contiguous() inside the loop)
    # x_c: [nspec, batch,      f_in]
    # y_c: [nspec, f_in,       f_out]
    x_c = torch.view_as_complex(x.contiguous())
    y_c = torch.view_as_complex(y.contiguous())

    # Output has the same spectral size as input (SO(3) -> SO(3))
    output = x.new_empty(nspec, nbatch, nfeature_out, 2)

    begin = 0
    for l in range(nl):
        L    = 2 * l + 1
        size = L ** 2
        s    = slice(begin, begin + size)

        # View the [L^2, ...] spectral block as [L, L, ...]
        # x: [m, p, batch, f_in]   m=output spectral, p=contracted spectral
        # y: [n, p, f_in, f_out]   n=output spectral, p=contracted spectral
        xc_l = x_c[s].view(L, L, nbatch,       nfeature_in)   # [m, p, batch, f_in]
        yc_l = y_c[s].view(L, L, nfeature_in,  nfeature_out)  # [n, p, f_in,  f_out]

        # Contract over p (spectral) and f_in simultaneously
        # z[m, n, batch, f_out] = sum_{p, f_in} x[m,p,b,i] * conj(y[n,p,i,f])
        zc = torch.einsum("mpbi, npif -> mnbf", xc_l, yc_l.conj())  # [m, n, batch, f_out]

        # Flatten [m, n] -> [m*n] and write into output buffer
        output[s] = torch.view_as_real(zc.reshape(size, nbatch, nfeature_out))

        begin += size

    assert begin == nspec
    return output   # [l*m*n, batch, f_out, 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     torch.manual_seed(42)

#     # nl=4 -> nspec = 4*(4*16-1)/3 = 4*63/3 = 84
#     nl           = 4
#     nspec        = nl * (4 * nl ** 2 - 1) // 3    # 84
#     nbatch       = 3
#     nfeature_in  = 5
#     nfeature_out = 7

#     print(f"so3_mm test  |  nl={nl}, nspec={nspec}")
#     print(f"  x: [{nspec}, {nbatch}, {nfeature_in}, 2]")
#     print(f"  y: [{nspec}, {nfeature_in}, {nfeature_out}, 2]")
#     print(f"  z: [{nspec}, {nbatch}, {nfeature_out}, 2]  (same nspec as input)")

#     x = torch.randn(nspec, nbatch,      nfeature_in,  2)
#     y = torch.randn(nspec, nfeature_in, nfeature_out, 2)

#     # --- Test 1: output shape ---
#     z = so3_mm(x, y)
#     assert tuple(z.shape) == (nspec, nbatch, nfeature_out, 2), \
#         f"Shape mismatch: {tuple(z.shape)}"
#     print(f"\nTest 1 -- output shape {tuple(z.shape)}  PASSED")

#     # --- Test 2: output nspec matches input nspec (SO3 -> SO3, no lifting) ---
#     assert z.size(0) == x.size(0), \
#         "so3_mm should preserve nspec (unlike s2_mm which lifts S2->SO3)"
#     print(f"Test 2 -- output nspec == input nspec ({nspec})  PASSED")

#     # --- Test 3: linearity in x ---
#     alpha = 2.5
#     x64  = x.double()
#     x264 = torch.randn_like(x64)
#     y64  = y.double()
#     z_sum    = so3_mm(x64 + alpha * x264, y64)
#     z_linear = so3_mm(x64, y64) + alpha * so3_mm(x264, y64)
#     err3 = (z_sum - z_linear).abs().max().item()
#     print(f"\nTest 3 -- linearity in x (float64):  max error {err3:.2e}  (target < 1e-10)")
#     assert err3 < 1e-10, f"FAILED: {err3:.2e}"
#     print("  PASSED")

#     x2f     = torch.randn_like(x)
#     z_sum_f    = so3_mm(x + alpha * x2f, y)
#     z_linear_f = so3_mm(x, y) + alpha * so3_mm(x2f, y)
#     err3f = (z_sum_f - z_linear_f).abs().max().item()
#     print(f"  float32 rounding error: {err3f:.2e}  (expected ~1e-5, not a bug)")
#     assert err3f < 1e-3, f"float32 error unexpectedly large: {err3f:.2e}"
#     print("  float32 check PASSED")

#     # --- Test 4: linearity in y ---
#     y264 = torch.randn_like(y64)
#     z_sum    = so3_mm(x64, y64 + alpha * y264)
#     z_linear = so3_mm(x64, y64) + alpha * so3_mm(x64, y264)
#     err4 = (z_sum - z_linear).abs().max().item()
#     print(f"\nTest 4 -- linearity in y (float64):  max error {err4:.2e}  (target < 1e-10)")
#     assert err4 < 1e-10, f"FAILED: {err4:.2e}"
#     print("  PASSED")

#     y2f = torch.randn_like(y)
#     z_sum_f    = so3_mm(x, y + alpha * y2f)
#     z_linear_f = so3_mm(x, y) + alpha * so3_mm(x, y2f)
#     err4f = (z_sum_f - z_linear_f).abs().max().item()
#     print(f"  float32 rounding error: {err4f:.2e}  (expected ~1e-5, not a bug)")
#     assert err4f < 1e-3, f"float32 error unexpectedly large: {err4f:.2e}"
#     print("  float32 check PASSED")

#     # --- Test 5: manual per-degree check for l=2 ---
#     # Manually compute degree l=2 block and compare with so3_mm output
#     l        = 2
#     L        = 2 * l + 1                             # 5
#     begin_l2 = sum((2 * k + 1) ** 2 for k in range(l))
#     s        = slice(begin_l2, begin_l2 + L * L)

#     x_c = torch.view_as_complex(x.contiguous())
#     y_c = torch.view_as_complex(y.contiguous())

#     xc_l = x_c[s].view(L, L, nbatch,      nfeature_in)
#     yc_l = y_c[s].view(L, L, nfeature_in, nfeature_out)
#     zc_ref = torch.einsum("mpbi, npif -> mnbf", xc_l, yc_l.conj())
#     zc_ref = zc_ref.reshape(L * L, nbatch, nfeature_out)

#     z_block = torch.view_as_complex(z[s].contiguous())
#     err5 = (z_block - zc_ref).abs().max().item()
#     print(f"\nTest 5 -- manual l=2 block check:  max error {err5:.2e}  (target < 1e-6)")
#     assert err5 < 1e-6, f"FAILED: {err5:.2e}"
#     print("  PASSED")

#     # --- Test 6: gradients flow ---
#     x_g = x.clone().requires_grad_(True)
#     y_g = y.clone().requires_grad_(True)
#     z_g = so3_mm(x_g, y_g)
#     z_g.sum().backward()
#     assert x_g.grad is not None and y_g.grad is not None, "Gradients did not flow"
#     print(f"\nTest 6 -- gradients flow through so3_mm  PASSED")

#     # --- Test 7: GPU == CPU (only if CUDA available) ---
#     if torch.cuda.is_available():
#         z_cpu = so3_mm(x, y)
#         z_gpu = so3_mm(x.cuda(), y.cuda()).cpu()
#         rel = (z_cpu - z_gpu).abs().max().item() / (z_cpu.std().item() + 1e-8)
#         print(f"\nTest 7 -- GPU vs CPU:  relative error {rel:.2e}  (target < 1e-4)")
#         assert rel < 1e-4, f"FAILED: {rel:.2e}"
#         print("  PASSED")
#     else:
#         print(f"\nTest 7 -- GPU vs CPU:  SKIPPED (no CUDA device)")

#     print("\nAll tests passed.")
