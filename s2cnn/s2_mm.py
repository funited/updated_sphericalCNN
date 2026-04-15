# @title
# s2_mm.py
# Updated for Python 3.12 / PyTorch 2.10
#
# Changes vs original:
#   - Removed all custom CUDA C kernel templates       -> torch.einsum on complex tensors
#   - Removed _cuda_S2_mm autograd class               -> no longer needed, autograd works through einsum
#   - Removed s2cnn.utils.cuda dependency              -> no longer needed
#   - Removed s2cnn.utils.complex.complex_mm           -> replaced with torch.view_as_complex + .conj()
#   - CPU and GPU handled by the same code path        -> PyTorch dispatches to CUDA automatically

import torch


def s2_mm(x, y):
    """
    Spherical convolution theorem: multiply signal and kernel spectral coefficients.

    For each harmonic degree l, computes an outer product over the m-indices
    (signal) and n-indices (kernel), contracting over feature_in.

    Mathematically for each l:
        z[m, n, batch, f_out] = sum_{f_in}  x[m, batch, f_in]  *  conj(y[n, f_in, f_out])

    The conjugate on y comes from the spherical convolution theorem:
    convolution in spatial domain = pointwise product of SHT coefficients,
    where one side is conjugated.

    :param x:      [l*m,     batch,      feature_in,  2]   signal   spectral coefficients
    :param y:      [l*m,     feature_in, feature_out, 2]   kernel   spectral coefficients
    :return:       [l*m*n,   batch,      feature_out, 2]   output   spectral coefficients
                   (the extra n dimension lifts S2 -> SO3)
    """
    assert x.size(-1) == 2,           "last dim of x must be 2 (re, im)"
    assert y.size(-1) == 2,           "last dim of y must be 2 (re, im)"
    assert x.size(0)  == y.size(0),   f"nspec mismatch: x={x.size(0)}, y={y.size(0)}"
    assert x.size(2)  == y.size(1),   f"feature_in mismatch: x={x.size(2)}, y={y.size(1)}"

    nspec        = x.size(0)
    nbatch       = x.size(1)
    nfeature_in  = x.size(2)
    nfeature_out = y.size(2)
    nl           = round(nspec ** 0.5)
    assert nl ** 2 == nspec, f"nspec ({nspec}) must be a perfect square (got nl={nl})"

    # Output size: sum_{l=0}^{nl-1} (2l+1)^2  =  nl*(4*nl^2 - 1) / 3
    nspec_out = nl * (4 * nl ** 2 - 1) // 3

    # Pre-allocate output  [l*m*n, batch, f_out, 2]
    output = x.new_empty(nspec_out, nbatch, nfeature_out, 2)

    # Convert full tensors to complex views once (avoids repeated contiguous() calls)
    # x_c: [nspec,  batch,       f_in]
    # y_c: [nspec,  f_in,        f_out]
    x_c = torch.view_as_complex(x.contiguous())
    y_c = torch.view_as_complex(y.contiguous())

    out_begin = 0
    for l in range(nl):
        L   = 2 * l + 1
        s   = slice(l ** 2, l ** 2 + L)    # spectral slice for degree l

        xc_l = x_c[s]                      # [L, batch,  f_in]   complex
        yc_l = y_c[s]                      # [L, f_in,   f_out]  complex

        # Outer product over m (signal) and n (kernel), contract over f_in:
        #   z[m, n, batch, f_out] = sum_{f_in} xc_l[m, batch, f_in] * conj(yc_l[n, f_in, f_out])
        zc = torch.einsum("mbi, nif -> mnbf", xc_l, yc_l.conj())   # [L, L, batch, f_out] complex

        # Flatten m*n and write into output buffer
        zc_flat = zc.reshape(L * L, nbatch, nfeature_out)           # [L^2, batch, f_out]
        output[out_begin : out_begin + L * L] = torch.view_as_real(zc_flat)

        out_begin += L * L

    assert out_begin == nspec_out, f"Output size mismatch: wrote {out_begin}, expected {nspec_out}"
    return output   # [l*m*n, batch, f_out, 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     torch.manual_seed(42)

#     # nl=4 -> nspec = 16 (l*m), nspec_out = 4*(64-1)/3 = 84 (l*m*n)
#     nl           = 4
#     nspec        = nl ** 2          # 16
#     nbatch       = 3
#     nfeature_in  = 5
#     nfeature_out = 7
#     nspec_out    = nl * (4 * nl ** 2 - 1) // 3   # 84

#     print(f"s2_mm test  |  nl={nl}, nspec={nspec}, nspec_out={nspec_out}")
#     print(f"  x: [{nspec}, {nbatch}, {nfeature_in}, 2]")
#     print(f"  y: [{nspec}, {nfeature_in}, {nfeature_out}, 2]")
#     print(f"  z: [{nspec_out}, {nbatch}, {nfeature_out}, 2]")

#     x = torch.randn(nspec,  nbatch,       nfeature_in,  2)
#     y = torch.randn(nspec,  nfeature_in,  nfeature_out, 2)

#     # --- Test 1: output shape ---
#     z = s2_mm(x, y)
#     assert tuple(z.shape) == (nspec_out, nbatch, nfeature_out, 2), \
#         f"Shape mismatch: {tuple(z.shape)}"
#     print(f"\nTest 1 -- output shape {tuple(z.shape)}  PASSED")

#     # --- Test 2: linearity in x ---
#     alpha = 2.5
#     x2    = torch.randn_like(x)
#     z_sum     = s2_mm(x + alpha * x2, y)
#     z_linear  = s2_mm(x, y) + alpha * s2_mm(x2, y)
#     err2 = (z_sum - z_linear).abs().max().item()
#     print(f"\nTest 2 -- linearity in x:  max error {err2:.2e}  (target < 1e-5)")
#     assert err2 < 1e-5, f"FAILED: {err2:.2e}"
#     print("  PASSED")

#     # --- Test 3: linearity in y ---
#     y2   = torch.randn_like(y)
#     z_sum    = s2_mm(x, y + alpha * y2)
#     z_linear = s2_mm(x, y) + alpha * s2_mm(x, y2)
#     err3 = (z_sum - z_linear).abs().max().item()
#     print(f"\nTest 3 -- linearity in y:  max error {err3:.2e}  (target < 1e-5)")
#     assert err3 < 1e-5, f"FAILED: {err3:.2e}"
#     print("  PASSED")

#     # --- Test 4: result is consistent per-degree ---
#     # Manually compute one degree (l=2) and compare with s2_mm output
#     l     = 2
#     L     = 2 * l + 1                        # 5
#     s     = slice(l ** 2, l ** 2 + L)        # spectral slice

#     x_c   = torch.view_as_complex(x.contiguous())
#     y_c   = torch.view_as_complex(y.contiguous())
#     zc_ref = torch.einsum("mbi, nif -> mnbf", x_c[s], y_c[s].conj())  # [L,L,batch,f_out]
#     zc_ref = zc_ref.reshape(L * L, nbatch, nfeature_out)

#     # Locate l=2 block in output: offset = sum_{k=0}^{l-1} (2k+1)^2
#     out_start = sum((2 * k + 1) ** 2 for k in range(l))
#     z_block   = torch.view_as_complex(z[out_start : out_start + L * L].contiguous())

#     err4 = (z_block - zc_ref).abs().max().item()
#     print(f"\nTest 4 -- manual l=2 block check:  max error {err4:.2e}  (target < 1e-6)")
#     assert err4 < 1e-6, f"FAILED: {err4:.2e}"
#     print("  PASSED")

#     # --- Test 5: gradients flow (autograd check) ---
#     x_g = x.clone().requires_grad_(True)
#     y_g = y.clone().requires_grad_(True)
#     z_g = s2_mm(x_g, y_g)
#     loss = z_g.sum()
#     loss.backward()
#     assert x_g.grad is not None and y_g.grad is not None, "Gradients did not flow"
#     print(f"\nTest 5 -- gradients flow through s2_mm  PASSED")

#     # --- Test 6: GPU == CPU  (only if CUDA available) ---
#     if torch.cuda.is_available():
#         z_cpu = s2_mm(x, y)
#         z_gpu = s2_mm(x.cuda(), y.cuda()).cpu()
#         rel_err = (z_cpu - z_gpu).abs().max().item() / (z_cpu.std().item() + 1e-8)
#         print(f"\nTest 6 -- GPU vs CPU:  relative error {rel_err:.2e}  (target < 1e-4)")
#         assert rel_err < 1e-4, f"FAILED: {rel_err:.2e}"
#         print("  PASSED")
#     else:
#         print(f"\nTest 6 -- GPU vs CPU:  SKIPPED (no CUDA device)")

#     print("\nAll tests passed.")
