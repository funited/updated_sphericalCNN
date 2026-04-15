# @title
# so3_rotation.py
# Updated for Python 3.12 / NumPy 2.0 / PyTorch 2.10
#
# Changes vs original:
#   - cached_dirpklgz          -> lru_cache (in-memory)
#   - device_type/device_index -> removed from lru_cache key (same pattern as all other files)
#   - complex_mm               -> torch.view_as_complex + torch.einsum
#   - .astype(np.complex64).view(np.float32) -> np.real/imag stack  (cleaner, no view tricks)
#   - Wigner-D matrices stored as float64 numpy, cast to target dtype at use time

from functools import lru_cache
import numpy as np
import torch

# from so3_fft import SO3_fft_real, SO3_ifft_real


def so3_rotation(x, alpha, beta, gamma):
    """
    Rotate a signal on SO(3) by the rotation specified by Euler angles (alpha, beta, gamma).

    Implements rotation in the spectral domain:
        1. Forward SO(3) FFT
        2. Multiply each degree-l block by the Wigner-D matrix D^l(alpha, beta, gamma)
        3. Inverse SO(3) FFT

    :param x:      [..., beta, alpha, gamma]   shape (..., 2b, 2b, 2b)
    :param alpha:  first  Euler angle (rotation around z-axis)
    :param beta:   second Euler angle (rotation around y-axis)
    :param gamma:  third  Euler angle (rotation around z-axis again)
    :return:       [..., beta, alpha, gamma]   same shape as input
    """
    b      = x.size(-1) // 2
    x_size = x.size()

    # Wigner-D matrices: list of length b, Us[l] is complex Tensor [L, L]
    Us = _get_so3_rotation(b, alpha, beta, gamma, device=x.device, dtype=x.dtype)

    # Forward SO(3) FFT: real [..., 2b, 2b, 2b] -> spectral [l*m*n, ..., 2]
    x = SO3_fft_real.apply(x)      # [l*m*n, ..., 2]

    # Apply rotation per degree l in spectral domain
    # For each l: Fz[m, batch] = sum_n  U[m, n] * conj(Fx[n, batch])
    # (conjugation convention matches original complex_mm(U, Fx, conj_x=True))
    out_list = []
    begin    = 0
    for l in range(b):
        L    = 2 * l + 1
        size = L ** 2

        # Spectral ordering is m-major (s = m*L + n), so view(L, L*rest)
        # correctly gives [m, n*batch_flat].
        Fx   = x[begin : begin + size]                           # [L^2, ..., 2]
        rest = Fx.numel() // (L * L * 2)                        # product of batch dims
        Fx_c = torch.view_as_complex(
            Fx.contiguous().view(L, L * rest, 2)
        )                                                         # [L, L*rest] complex

        U = Us[l]                                                # [L, L] complex

        # Original: complex_mm(U, Fx, conj_x=True) conjugates the FIRST arg (U).
        # conj(U) @ Fx -- for identity U=I: conj(I)@Fx = Fx (correct)
        # NOT U @ conj(Fx): that gives conj(Fx) for U=I (wrong).
        Fz_c = torch.einsum("mk, kb -> mb", U.conj(), Fx_c)     # [L, L*rest]

        out_list.append(torch.view_as_real(Fz_c).view(size, rest, 2))

        begin += size

    Fz = torch.cat(out_list, dim=0)                    # [l*m*n, rest, 2]
    # Restore full batch shape before inverse FFT
    Fz = Fz.view(Fz.size(0), *x_size[:-3], 2)

    # Inverse SO(3) FFT: spectral -> real [..., 2b, 2b, 2b]
    z = SO3_ifft_real.apply(Fz)
    return z.contiguous().view(*x_size)


# ---------------------------------------------------------------------------
# Wigner-D matrix helpers
# ---------------------------------------------------------------------------

def _get_so3_rotation(b: int, alpha: float, beta: float, gamma: float,
                      device: torch.device, dtype: torch.dtype):
    """
    Return list of Wigner-D matrices [D^0, D^1, ..., D^{b-1}] as complex Tensors on `device`.
    Us[l] has shape [2l+1, 2l+1].

    numpy arrays cached separately (device-agnostic),
    moved to device/dtype here without polluting the lru_cache key.
    """
    Us_np = _compute_so3_rotation(b, alpha, beta, gamma)   # list of complex128 arrays
    return [
        torch.tensor(U, dtype=_complex_dtype(dtype), device=device)
        for U in Us_np
    ]


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Map a real dtype to its complex counterpart."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


@lru_cache(maxsize=128)
def _compute_so3_rotation(b: int, alpha: float, beta: float, gamma: float):
    """
    Compute Wigner-D matrices D^l(alpha, beta, gamma) for l = 0..b-1.

    Returns a tuple of complex128 ndarrays, Us[l] has shape [(2l+1), (2l+1)].

    Us[l][m, n] = exp(i*m*alpha) * d^l_{mn}(beta) * exp(i*n*gamma)

    The result is cached keyed on (b, alpha, beta, gamma) — all Python scalars,
    safely hashable by lru_cache.

    Requires: pip install git+https://github.com/AMLab-Amsterdam/lie_learn
    """
    try:
        from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    except ImportError as e:
        raise ImportError(
            "lie_learn is required.\n"
            "Install: pip install git+https://github.com/AMLab-Amsterdam/lie_learn"
        ) from e

    Us = []
    for l in range(b):
        D = wigner_D_matrix(
            l, alpha, beta, gamma,
            field='complex',
            normalization='quantum',
            order='centered',
            condon_shortley='cs'
        )                                              # complex128 [(2l+1), (2l+1)]
        Us.append(D.astype(np.complex128))

    return tuple(Us)                                   # tuple so lru_cache can hash it


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import math
#     # from so3_fft import so3_rifft, _nspec
#     torch.manual_seed(42)

#     b = 4
#     print(f"so3_rotation test  |  b={b},  grid={2*b}^3")

#     # Generate a BANDLIMITED signal via rifft of random spectral coefficients.
#     # A plain torch.randn spatial signal is NOT bandlimited to degree b-1:
#     # so3_rotation does rfft -> (spectral multiply) -> rifft, which only
#     # roundtrips exactly on bandlimited signals. Using a non-bandlimited
#     # signal would discard high-freq energy and make identity-rotation fail
#     # with large errors that are expected, not bugs.
#     nsp  = _nspec(b)
#     spec = torch.randn(nsp, 2, 3, 2)          # [nsp, batch, channel, 2]
#     x    = so3_rifft(spec, b_out=b)           # [2, 3, 2b, 2b, 2b]  bandlimited

#     # --- Test 1: output shape preserved ---
#     y = so3_rotation(x, alpha=0.3, beta=0.5, gamma=0.7)
#     assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
#     print(f"\nTest 1 -- output shape preserved {tuple(y.shape)}  PASSED")

#     # --- Test 2: identity rotation on bandlimited signal ---
#     y_id = so3_rotation(x, alpha=0.0, beta=0.0, gamma=0.0)
#     err2 = (x - y_id).abs().max().item()
#     print(f"\nTest 2 -- identity rotation (bandlimited):  max error {err2:.2e}  (target < 1e-4)")
#     assert err2 < 1e-4, f"FAILED: {err2:.2e}"
#     print("  PASSED")

#     # --- Test 3: rotation is norm-preserving (unitary) ---
#     norm_x = x.pow(2).sum().item()
#     norm_y = y.pow(2).sum().item()
#     rel    = abs(norm_x - norm_y) / norm_x
#     print(f"\nTest 3 -- norm preserving:  rel error {rel:.2e}  (target < 5e-2)")
#     assert rel < 5e-2, f"FAILED: {rel:.2e}"
#     print("  PASSED")

#     # --- Test 4: rotation composition R(a)*R(a) == R(2a) ---
#     alpha = math.pi / 6
#     y1   = so3_rotation(x,  alpha=alpha,     beta=0.0, gamma=0.0)
#     y2   = so3_rotation(y1, alpha=alpha,     beta=0.0, gamma=0.0)
#     y_2a = so3_rotation(x,  alpha=2 * alpha, beta=0.0, gamma=0.0)
#     err4 = (y2 - y_2a).abs().max().item()
#     print(f"\nTest 4 -- composition R(a)*R(a)==R(2a):  max error {err4:.2e}  (target < 1e-3)")
#     assert err4 < 1e-3, f"FAILED: {err4:.2e}"
#     print("  PASSED")

#     # --- Test 5: gradients flow ---
#     x_g = x.clone().requires_grad_(True)
#     so3_rotation(x_g, alpha=0.3, beta=0.5, gamma=0.7).sum().backward()
#     assert x_g.grad is not None, "Gradient did not flow"
#     print(f"\nTest 5 -- gradients flow  PASSED")

#     print("\nAll tests passed.")
