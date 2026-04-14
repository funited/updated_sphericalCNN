# @title
# s2_fft.py
# Updated for Python 3.12 / NumPy 2.0 / PyTorch 2.10
#
# Changes vs original:
#   - Removed cached_dirpklgz        → lru_cache (in-memory)
#   - Removed custom CUDA C kernels  → torch.einsum (PyTorch handles GPU internally)
#   - np.float                       → np.float64  (removed in NumPy 1.24)
#   - device excluded from lru_cache key (not safely hashable across calls)
#   - Wigner-d computed in float64, cast to float32 for GPU ops

from functools import lru_cache
import numpy as np
import torch
import torch.fft


# ---------------------------------------------------------------------------
# Public API: forward and inverse S2 FFT
# ---------------------------------------------------------------------------

def s2_fft(x, for_grad=False, b_out=None):
    """
    Forward Spherical Harmonic Transform (real -> spectral).

    :param x:      [..., beta, alpha, complex]   last dim = 2 (re, im)
    :param b_out:  output bandwidth  <= b_in  (truncates high-freq modes)
    :return:       [l*m, ..., complex]
    """
    assert x.size(-1) == 2,        "last dim must be 2 (re, im)"
    b_in = x.size(-2) // 2
    assert x.size(-2) == 2 * b_in, f"beta dim must be 2*b_in={2*b_in}, got {x.size(-2)}"
    assert x.size(-3) == 2 * b_in, f"alpha dim must be 2*b_in={2*b_in}, got {x.size(-3)}"
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in, f"b_out ({b_out}) must be <= b_in ({b_in})"

    batch_size = x.size()[:-3]                          # e.g. (B, C)
    nbatch     = x.numel() // (2 * b_in * 2 * b_in * 2)
    nspec      = b_out ** 2

    x = x.view(nbatch, 2 * b_in, 2 * b_in, 2)         # [B, beta, alpha, 2]

    # Wigner matrix [2*b_in, nspec], on same device/dtype as x
    wigner = _get_wigner(b_in, nl=b_out, weighted=not for_grad, device=x.device, dtype=x.dtype)

    # FFT along alpha axis: [B, beta, m, 2]
    x = torch.view_as_real(torch.fft.fft(torch.view_as_complex(x.contiguous()), dim=2))

    # Accumulate over beta via Wigner-d weights -> [nspec, B, 2]
    output = x.new_zeros(nspec, nbatch, 2)
    for l in range(b_out):
        s  = slice(l ** 2, l ** 2 + 2 * l + 1)
        # Reorder alpha-FFT bins to centred-m order: [-l,..,0,..,+l]
        xx = torch.cat((x[:, :, -l:], x[:, :, :l + 1]), dim=2) if l > 0 else x[:, :, :1]
        # xx: [B, beta, 2l+1, 2],  wigner[:,s]: [beta, 2l+1]
        output[s] = torch.einsum("bm, zbmc -> mzc", wigner[:, s], xx)

    return output.view(-1, *batch_size, 2)              # [l*m, ..., 2]


def s2_ifft(x, for_grad=False, b_out=None):
    """
    Inverse Spherical Harmonic Transform (spectral -> spatial).

    :param x:      [l*m, ..., complex]   last dim = 2 (re, im)
    :param b_out:  output spatial bandwidth  >= b_in
    :return:       [..., beta, alpha, complex]
    """
    assert x.size(-1) == 2, "last dim must be 2 (re, im)"
    nspec = x.size(0)
    b_in  = round(nspec ** 0.5)
    assert nspec == b_in ** 2, f"nspec ({nspec}) must be a perfect square"
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in, f"b_out ({b_out}) must be >= b_in ({b_in})"

    batch_size = x.size()[1:-1]                         # e.g. (B, C)
    nbatch     = x.numel() // (nspec * 2)

    x = x.view(nspec, nbatch, 2)                        # [nspec, B, 2]

    # Wigner matrix [2*b_out, nspec]
    wigner = _get_wigner(b_out, nl=b_in, weighted=for_grad, device=x.device, dtype=x.dtype)

    # Accumulate spectral -> [B, beta, alpha-m, 2]
    output = x.new_zeros(nbatch, 2 * b_out, 2 * b_out, 2)
    for l in range(b_in):
        s   = slice(l ** 2, l ** 2 + 2 * l + 1)
        out = torch.einsum("mzc, bm -> zbmc", x[s], wigner[:, s])
        # out: [B, beta, 2l+1, 2] -- scatter back to FFT bin order
        output[:, :, :l + 1]  += out[:, :, -l - 1:]   # m = 0 .. +l
        if l > 0:
            output[:, :, -l:] += out[:, :, :l]         # m = -l .. -1

    # iFFT along alpha, undo PyTorch's 1/N normalisation
    n_alpha = output.size(2)                            # = 2 * b_out
    output  = torch.view_as_real(
        torch.fft.ifft(torch.view_as_complex(output.contiguous()), dim=2)
    ) * n_alpha

    return output.view(*batch_size, 2 * b_out, 2 * b_out, 2)


# ---------------------------------------------------------------------------
# Wigner-d matrix helpers
# ---------------------------------------------------------------------------

def _get_wigner(b: int, nl: int, weighted: bool,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return Wigner-d matrix as a Tensor on `device`.
    Shape: [2*b, nl^2]

    The numpy array is cached separately (device/dtype-agnostic),
    then moved to the correct device/dtype here.
    This keeps lru_cache keys simple and avoids device-hashability issues.
    """
    dss = _compute_wigner_d(b, nl, weighted)    # float64 numpy [2*b, nl^2]
    return torch.tensor(dss, dtype=dtype, device=device).contiguous()


@lru_cache(maxsize=64)
def _compute_wigner_d(b: int, nl: int, weighted: bool) -> np.ndarray:
    """
    Compute Wigner-d evaluation matrix for the SOFT beta quadrature grid.

    Returns float64 ndarray of shape [2*b, nl^2].

    Each row  = one SOFT quadrature point (beta angle).
    Each col  = one (l, m) mode; specifically d^l_{m,0}(beta),
                which is the beta-dependent part of Y_l^m.

    weighted=True  : multiply by quadrature weights  (used in forward SHT)
    weighted=False : multiply by (2l+1)              (used in inverse SHT)

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

    # SOFT quadrature beta points: strictly between 0 and pi (never at poles)
    betas = (np.arange(2 * b, dtype=np.float64) + 0.5) / (2 * b) * np.pi

    # Quadrature weights that approximate integral: sum_i w_i f(beta_i) ~ integral
    w = S3.quadrature_weights(b) * 2 * b           # shape [2*b]
    assert w.shape[0] == len(betas)

    rows = []
    for beta_idx, beta in enumerate(betas):
        cols = []
        for l in range(nl):
            # Small Wigner-d matrix d^l_{mn}(beta) is always real-valued.
            # We take the n=0 column (index l in centered order: -l..0..+l).
            d = wigner_d_matrix(
                l, beta,
                field='complex',
                normalization='quantum',
                order='centered',
                condon_shortley='cs'
            )
            col = np.real(d[:, l]).astype(np.float64)   # shape [2l+1]
            col = col * (w[beta_idx] if weighted else float(2 * l + 1))
            cols.append(col)

        rows.append(np.concatenate(cols))               # [nl^2]

    return np.stack(rows)                               # [2*b, nl^2]


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers (real-valued interface)
# ---------------------------------------------------------------------------

class S2_fft_real(torch.autograd.Function):
    """
    Forward:  real [..., beta, alpha]     ->  complex [l*m, ..., 2]
    Backward: adjoint iSHT
    """
    @staticmethod
    def forward(ctx, x, b_out=None):
        ctx.b_out = b_out
        ctx.b_in  = x.size(-1) // 2          # alpha dim = 2*b_in
        # Promote real spatial signal to complex (zero imaginary part)
        xc = torch.stack([x, torch.zeros_like(x)], dim=-1)
        return s2_fft(xc, b_out=ctx.b_out)

    @staticmethod
    def backward(ctx, grad_output):
        x_grad = s2_ifft(grad_output, for_grad=True, b_out=ctx.b_in)
        return x_grad[..., 0], None


class S2_ifft_real(torch.autograd.Function):
    """
    Forward:  complex [l*m, ..., 2]       ->  real [..., beta, alpha]
    Backward: adjoint SHT
    """
    @staticmethod
    def forward(ctx, x, b_out=None):
        nspec     = x.size(0)
        ctx.b_out = b_out
        ctx.b_in  = round(nspec ** 0.5)
        return s2_ifft(x, b_out=ctx.b_out)[..., 0]     # take real part

    @staticmethod
    def backward(ctx, grad_output):
        xc = torch.stack([grad_output, torch.zeros_like(grad_output)], dim=-1)
        return s2_fft(xc, for_grad=True, b_out=ctx.b_in), None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     torch.manual_seed(42)
#     b = 8

#     print(f"Testing S2 FFT  |  bandwidth b={b}")
#     print(f"  Spatial grid  : {2*b} x {2*b} = {4*b**2} points")
#     print(f"  Spectral modes: {b**2}")
#     print()

#     # ------------------------------------------------------------------
#     # WHY we test spectral->spatial->spectral (not spatial->spectral->spatial):
#     #
#     #   torch.rand on the spatial grid is NOT bandlimited to degree b-1,
#     #   so SHT followed by iSHT only recovers the low-frequency projection
#     #   -- the error would be huge by design, not a bug.
#     #
#     #   Starting from spectral coefficients guarantees the signal IS
#     #   bandlimited to degree b-1, so the round-trip must be exact
#     #   (up to floating-point rounding).
#     # ------------------------------------------------------------------

#     # Test 1: internal s2_fft / s2_ifft functions
#     nspec      = b ** 2
#     spec_orig  = torch.randn(nspec, 2, 3, 2, dtype=torch.float32)  # [l*m, B, C, 2]
#     spatial    = s2_ifft(spec_orig, b_out=b)                        # [..., beta, alpha, 2]
#     spec_back  = s2_fft(spatial,   b_out=b)                        # [l*m, ..., 2]

#     err1 = (spec_orig - spec_back).abs().max().item()
#     print(f"Test 1 -- s2_fft/s2_ifft spectral->spatial->spectral")
#     print(f"  Max error: {err1:.2e}  (target < 1e-4)")
#     assert err1 < 1e-4, f"FAILED: error={err1:.2e}"
#     print("  PASSED")

#     # Test 2: autograd wrappers S2_fft_real / S2_ifft_real
#     #   Generate a bandlimited real spatial signal: take real part of iSHT
#     #   of random complex spectral coefficients. Then verify spatial->spectral->spatial.
#     spec_seed    = torch.randn(nspec, 1, 2, dtype=torch.float32)    # [l*m, 1, 2]
#     spatial_real = s2_ifft(spec_seed, b_out=b)[..., 0]              # [1, 2b, 2b]  real
#     spec2        = S2_fft_real.apply(spatial_real, b)               # [l*m, 1, 2]
#     back2        = S2_ifft_real.apply(spec2, b)                     # [1, 2b, 2b]

#     err2 = (spatial_real - back2).abs().max().item()
#     print(f"\nTest 2 -- S2_fft_real/S2_ifft_real spatial->spectral->spatial (bandlimited input)")
#     print(f"  Max error: {err2:.2e}  (target < 1e-4)")
#     assert err2 < 1e-4, f"FAILED: error={err2:.2e}"
#     print("  PASSED")

#     print("\nAll tests passed.")
