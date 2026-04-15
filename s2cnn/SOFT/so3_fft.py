# @title
# so3_fft.py
# Updated for Python 3.12 / NumPy 2.0 / PyTorch 2.10
#
# Changes vs original:
#   - cached_dirpklgz          -> lru_cache (in-memory, no fragile pickle files)
#   - Custom CUDA kernels       -> always use einsum path (PyTorch dispatches to CUDA)
#   - device in lru_cache key  -> removed; numpy array cached separately, moved to device in _get_wigner
#   - Wigner-d dtype            -> computed in float64, cast to target dtype in _get_wigner
#
# Relationship to s2_fft.py:
#   s2_fft  operates on S2:  2 spatial dims (beta, alpha)     1D FFT over alpha
#   so3_fft operates on SO3: 3 spatial dims (beta, alpha, gamma)  2D FFT over (alpha, gamma)
#
# The Wigner matrix here uses the FULL wigner_d matrix reshaped to (2l+1)^2
# (all m*n pairs), unlike s2_fft which only takes the n=0 column.

from functools import lru_cache
import numpy as np
import torch
import torch.fft


# ---------------------------------------------------------------------------
# Spectral index helpers
# ---------------------------------------------------------------------------

def _spec_slice(l):
    """Slice into the spectral buffer for degree l.  Size = (2l+1)^2."""
    start = l * (4 * l ** 2 - 1) // 3
    return slice(start, start + (2 * l + 1) ** 2)


def _nspec(b):
    """Total number of SO(3) spectral coefficients for bandwidth b."""
    return b * (4 * b ** 2 - 1) // 3


# ---------------------------------------------------------------------------
# Forward SO(3) FFT  (complex input)
# ---------------------------------------------------------------------------

def so3_fft(x, for_grad=False, b_out=None):
    """
    Forward SO(3) Fourier Transform.

    :param x:      [..., beta, alpha, gamma, 2]   last dim = complex (re, im)
    :param b_out:  output bandwidth (<= b_in)
    :return:       [l*m*n, ..., 2]
    """
    assert x.size(-1) == 2,         "last dim must be 2 (re, im)"
    b_in = x.size(-2) // 2
    assert x.size(-2) == 2 * b_in
    assert x.size(-3) == 2 * b_in
    assert x.size(-4) == 2 * b_in
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in, f"b_out ({b_out}) must be <= b_in ({b_in})"

    batch_size = x.size()[:-4]
    nbatch     = x.numel() // (2 * b_in * 2 * b_in * 2 * b_in * 2)
    nsp        = _nspec(b_out)

    x = x.view(nbatch, 2 * b_in, 2 * b_in, 2 * b_in, 2)   # [B, beta, alpha, gamma, 2]

    # Wigner matrix [2*b_in, nspec]
    wigner = _get_wigner(b_in, nl=b_out, weighted=not for_grad, device=x.device, dtype=x.dtype)

    # 2D FFT over (alpha, gamma): [B, beta, m, n, 2]
    x = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(x.contiguous()), dim=[2, 3]))

    # Accumulate over beta via Wigner-d weights -> [nsp, B, 2]
    output = x.new_zeros(nsp, nbatch, 2)
    for l in range(b_out):
        s  = _spec_slice(l)
        L  = 2 * l + 1
        l1 = min(l, b_in - 1)    # frequencies above b_in are zero (aliasing guard)

        # Gather relevant (m, n) bins into centred [L, L] window
        xx = x.new_zeros(nbatch, 2 * b_in, L, L, 2)
        xx[:, :, l:l + l1 + 1, l:l + l1 + 1]   = x[:, :, :l1 + 1,  :l1 + 1]
        if l1 > 0:
            xx[:, :, l - l1:l,     l:l + l1 + 1] = x[:, :, -l1:,     :l1 + 1]
            xx[:, :, l:l + l1 + 1, l - l1:l]     = x[:, :, :l1 + 1,  -l1:]
            xx[:, :, l - l1:l,     l - l1:l]      = x[:, :, -l1:,     -l1:]

        # wigner[:, s]: [2*b_in, L^2]  -> reshape to [2*b_in, L, L]
        wig_l = wigner[:, s].view(2 * b_in, L, L)

        # Contract over beta: [L, L, B, 2]
        out = torch.einsum("bmn, zbmnc -> mnzc", wig_l, xx)
        output[s] = out.reshape(L * L, nbatch, 2)

    return output.view(-1, *batch_size, 2)   # [l*m*n, ..., 2]


# ---------------------------------------------------------------------------
# Forward SO(3) FFT  (real input)
# ---------------------------------------------------------------------------

def so3_rfft(x, for_grad=False, b_out=None):
    """
    Forward SO(3) Fourier Transform for real-valued input signals.

    :param x:      [..., beta, alpha, gamma]   real-valued
    :param b_out:  output bandwidth
    :return:       [l*m*n, ..., 2]
    """
    b_in = x.size(-1) // 2
    assert x.size(-1) == 2 * b_in
    assert x.size(-2) == 2 * b_in
    assert x.size(-3) == 2 * b_in
    if b_out is None:
        b_out = b_in

    # Promote to complex by stacking a zero imaginary part, then call so3_fft
    xc = torch.stack([x, torch.zeros_like(x)], dim=-1)
    return so3_fft(xc, for_grad=for_grad, b_out=b_out)


# ---------------------------------------------------------------------------
# Inverse SO(3) FFT  (complex output)
# ---------------------------------------------------------------------------

def so3_ifft(x, for_grad=False, b_out=None):
    """
    Inverse SO(3) Fourier Transform.

    :param x:      [l*m*n, ..., 2]   spectral coefficients
    :param b_out:  output spatial bandwidth (>= b_in)
    :return:       [..., beta, alpha, gamma, 2]
    """
    assert x.size(-1) == 2
    nsp  = x.size(0)
    b_in = round((3 / 4 * nsp) ** (1 / 3))
    assert nsp == _nspec(b_in), f"nspec={nsp} is not valid for any integer b_in"
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in, f"b_out ({b_out}) must be >= b_in ({b_in})"

    batch_size = x.size()[1:-1]
    nbatch     = x.numel() // (nsp * 2)

    x = x.view(nsp, nbatch, 2)    # [nsp, B, 2]

    # Wigner matrix [2*b_out, nspec]
    wigner = _get_wigner(b_out, nl=b_in, weighted=for_grad, device=x.device, dtype=x.dtype)

    # Accumulate spectral -> [B, beta, alpha-m, gamma-n, 2]
    output = x.new_zeros(nbatch, 2 * b_out, 2 * b_out, 2 * b_out, 2)
    for l in range(min(b_in, b_out)):
        s  = _spec_slice(l)
        L  = 2 * l + 1
        l1 = min(l, b_out - 1)   # guard for b_out < b_in

        wig_l = wigner[:, s].view(2 * b_out, L, L)
        # out: [B, beta, L, L, 2]
        out = torch.einsum("mnzc, bmn -> zbmnc", x[s].view(L, L, nbatch, 2), wig_l)

        output[:, :, :l1 + 1,  :l1 + 1]  += out[:, :, l:l + l1 + 1, l:l + l1 + 1]
        if l > 0:
            output[:, :, -l1:,     :l1 + 1]  += out[:, :, l - l1:l,     l:l + l1 + 1]
            output[:, :, :l1 + 1,  -l1:]     += out[:, :, l:l + l1 + 1, l - l1:l]
            output[:, :, -l1:,     -l1:]      += out[:, :, l - l1:l,     l - l1:l]

    # 2D iFFT over (alpha, gamma), undo PyTorch's 1/N^2 normalisation
    n_ag = output.size(2)                              # = 2 * b_out
    output = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(output.contiguous()), dim=[2, 3])
    ) * (n_ag ** 2)

    return output.view(*batch_size, 2 * b_out, 2 * b_out, 2 * b_out, 2)


# ---------------------------------------------------------------------------
# Inverse SO(3) FFT  (real output)
# ---------------------------------------------------------------------------

def so3_rifft(x, for_grad=False, b_out=None):
    """
    Inverse SO(3) Fourier Transform, returning only the real part.

    :param x:      [l*m*n, ..., 2]   spectral coefficients
    :param b_out:  output spatial bandwidth
    :return:       [..., beta, alpha, gamma]   real-valued
    """
    out = so3_ifft(x, for_grad=for_grad, b_out=b_out)
    return out[..., 0].contiguous()


# ---------------------------------------------------------------------------
# Wigner-d matrix helpers  (same split-cache pattern as s2_fft.py)
# ---------------------------------------------------------------------------

def _get_wigner(b: int, nl: int, weighted: bool,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return cached Wigner-d matrix as a Tensor on `device`.
    Shape: [2*b, nspec]   where nspec = nl*(4*nl^2-1)//3

    numpy array cached separately (device-agnostic),
    moved to device/dtype here without polluting the lru_cache key.
    """
    dss = _compute_wigner_d(b, nl, weighted)    # float64 numpy [2*b, nspec]
    return torch.tensor(dss, dtype=dtype, device=device).contiguous()


@lru_cache(maxsize=32)
def _compute_wigner_d(b: int, nl: int, weighted: bool) -> np.ndarray:
    """
    Compute SO(3) Wigner-d evaluation matrix for the SOFT beta grid.

    Returns float64 ndarray of shape [2*b, nspec]
    where nspec = nl * (4*nl^2 - 1) // 3.

    Each row  = one SOFT quadrature point (beta angle).
    Each col  = one (l, m, n) Wigner-d coefficient:
                d^l_{mn}(beta), ALL m and n pairs flattened.

    This differs from s2_fft which only takes the n=0 column.
    Here the full (2l+1)x(2l+1) matrix is taken and reshaped.

    Requires: pip install git+https://github.com/AMLab-Amsterdam/lie_learn
    """
    try:
        from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
        import lie_learn.spaces.S3 as S3
    except ImportError as e:
        raise ImportError(
            "lie_learn is required.\n"
            "Install: pip install git+https://github.com/AMLab-Amsterdam/lie_learn"
        ) from e

    # SOFT quadrature beta points: strictly between 0 and pi
    betas = (np.arange(2 * b, dtype=np.float64) + 0.5) / (2 * b) * np.pi

    # Quadrature weights
    w = S3.quadrature_weights(b)                # shape [2*b]
    assert w.shape[0] == len(betas)

    rows = []
    for beta_idx, beta in enumerate(betas):
        cols = []
        for l in range(nl):
            # Full (2l+1)x(2l+1) small Wigner-d matrix at this beta
            d = wigner_d_matrix(
                l, beta,
                field='complex',
                normalization='quantum',
                order='centered',
                condon_shortley='cs'
            )                                   # shape [(2l+1), (2l+1)]

            d = np.real(d).astype(np.float64).reshape((2 * l + 1) ** 2)
            d = d * (w[beta_idx] if weighted else float(2 * l + 1))
            cols.append(d)

        rows.append(np.concatenate(cols))       # [nspec]

    return np.stack(rows)                       # [2*b, nspec]


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers (real-valued interface)
# ---------------------------------------------------------------------------

class SO3_fft_real(torch.autograd.Function):
    """
    Forward:  real [..., beta, alpha, gamma]     ->  complex [l*m*n, ..., 2]
    Backward: adjoint iSO3FFT
    """
    @staticmethod
    def forward(ctx, x, b_out=None):
        ctx.b_out = b_out
        ctx.b_in  = x.size(-1) // 2
        return so3_rfft(x, b_out=ctx.b_out)

    @staticmethod
    def backward(ctx, grad_output):
        # ifft of grad is not necessarily real -> use so3_ifft not so3_rifft
        return so3_ifft(grad_output, for_grad=True, b_out=ctx.b_in)[..., 0], None


class SO3_ifft_real(torch.autograd.Function):
    """
    Forward:  complex [l*m*n, ..., 2]           ->  real [..., beta, alpha, gamma]
    Backward: adjoint SO3FFT
    """
    @staticmethod
    def forward(ctx, x, b_out=None):
        nspec     = x.size(0)
        ctx.b_out = b_out
        ctx.b_in  = round((3 / 4 * nspec) ** (1 / 3))
        return so3_rifft(x, b_out=ctx.b_out)

    @staticmethod
    def backward(ctx, grad_output):
        return so3_rfft(grad_output, for_grad=True, b_out=ctx.b_in), None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     torch.manual_seed(42)
#     b = 5

#     nsp = _nspec(b)
#     print(f"SO3 FFT test  |  b={b},  spatial grid={2*b}^3={8*b**3},  nspec={nsp}")

#     # ------------------------------------------------------------------
#     # Test 1: spectral -> spatial -> spectral round-trip
#     # (Same reasoning as s2_fft.py: start from spectral to guarantee
#     #  the signal is bandlimited, so the round-trip must be exact.)
#     # ------------------------------------------------------------------
#     spec_orig = torch.randn(nsp, 2, 3, 2, dtype=torch.float32)   # [nsp, B, C, 2]
#     spatial   = so3_ifft(spec_orig, b_out=b)                     # [..., beta, alpha, gamma, 2]
#     spec_back = so3_fft(spatial,   b_out=b)                      # [nsp, ..., 2]

#     err1 = (spec_orig - spec_back).abs().max().item()
#     print(f"\nTest 1 -- so3_fft/so3_ifft spectral->spatial->spectral round-trip")
#     print(f"  Max error: {err1:.2e}  (target < 1e-4)")
#     assert err1 < 1e-4, f"FAILED: {err1:.2e}"
#     print("  PASSED")

#     # ------------------------------------------------------------------
#     # Test 2: real-valued round-trip via autograd wrappers
#     # ------------------------------------------------------------------
#     spec_seed    = torch.randn(nsp, 1, 2, dtype=torch.float32)
#     spatial_real = so3_rifft(spec_seed, b_out=b)                 # [1, 2b, 2b, 2b] real
#     spec2        = SO3_fft_real.apply(spatial_real, b)           # [nsp, 1, 2]
#     back2        = SO3_ifft_real.apply(spec2, b)                 # [1, 2b, 2b, 2b]

#     err2 = (spatial_real - back2).abs().max().item()
#     print(f"\nTest 2 -- SO3_fft_real/SO3_ifft_real spatial->spectral->spatial")
#     print(f"  Max error: {err2:.2e}  (target < 1e-4)")
#     assert err2 < 1e-4, f"FAILED: {err2:.2e}"
#     print("  PASSED")

#     # ------------------------------------------------------------------
#     # Test 3: gradients flow through SO3_fft_real / SO3_ifft_real
#     # ------------------------------------------------------------------
#     x_g = spatial_real.clone().requires_grad_(True)
#     spec_g = SO3_fft_real.apply(x_g, b)
#     spec_g.sum().backward()
#     assert x_g.grad is not None, "Gradient did not flow through SO3_fft_real"
#     print(f"\nTest 3 -- gradients flow through SO3_fft_real  PASSED")

#     spec_in = spec2.detach().requires_grad_(True)
#     out_g   = SO3_ifft_real.apply(spec_in, b)
#     out_g.sum().backward()
#     assert spec_in.grad is not None, "Gradient did not flow through SO3_ifft_real"
#     print(f"Test 3 -- gradients flow through SO3_ifft_real  PASSED")

#     # ------------------------------------------------------------------
#     # Test 4: so3_rfft == so3_fft on complex-promoted input
#     # ------------------------------------------------------------------
#     x_real = spatial_real[0]   # [2b, 2b, 2b]  real
#     xc     = torch.stack([x_real, torch.zeros_like(x_real)], dim=-1)  # complex

#     out_rfft = so3_rfft(x_real.unsqueeze(0), b_out=b)
#     out_fft  = so3_fft(xc.unsqueeze(0), b_out=b)

#     err4 = (out_rfft - out_fft).abs().max().item()
#     print(f"\nTest 4 -- so3_rfft == so3_fft on real input:  max error {err4:.2e}  (target < 1e-6)")
#     assert err4 < 1e-6, f"FAILED: {err4:.2e}"
#     print("  PASSED")

#     # ------------------------------------------------------------------
#     # Test 5: GPU == CPU (if CUDA available)
#     # ------------------------------------------------------------------
#     if torch.cuda.is_available():
#         spec_cpu = so3_ifft(spec_orig, b_out=b)
#         spec_gpu = so3_ifft(spec_orig.cuda(), b_out=b).cpu()
#         rel = (spec_cpu - spec_gpu).abs().max().item() / (spec_cpu.std().item() + 1e-8)
#         print(f"\nTest 5 -- GPU vs CPU (so3_ifft):  relative error {rel:.2e}  (target < 1e-4)")
#         assert rel < 1e-4, f"FAILED: {rel:.2e}"
#         print("  PASSED")
#     else:
#         print(f"\nTest 5 -- GPU vs CPU:  SKIPPED (no CUDA device)")

#     print("\nAll tests passed.")
