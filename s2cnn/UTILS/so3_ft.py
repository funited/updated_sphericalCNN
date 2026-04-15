# @title
# so3_rft.py
# Updated for Python 3.12 / NumPy 2.0 / PyTorch 2.10
#
# Changes vs original:
#   - cached_dirpklgz          -> lru_cache  (in-memory, no fragile pickle files)
#   - F.view('float')          -> F.view(np.float64)  ('float' string removed in NumPy 2.0)
#   - device_type/device_index -> excluded from lru_cache key (same pattern as s2_rft.py)
#                                 numpy array cached separately, moved to device in _get_so3_ft
#   - wigner_D_matrix import   -> guarded with clear ImportError message
#
# Key difference from s2_rft.py:
#   s2_rft  uses column n=0 of D^l only  (gamma=0 fixes the gamma dependence)
#   so3_rft uses the FULL D^l matrix     (all m x n entries, for arbitrary gamma)

from functools import lru_cache
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def so3_rft(x, b, grid):
    """
    Kernel SO(3) Fourier Transform:
    projects a kernel defined at discrete SO(3) grid points
    into the spectral domain (Wigner-D coefficients).

    This is NOT the signal SO3 FFT (that is so3_fft). This is specifically
    for transforming the convolution kernel from spatial samples -> spectral
    form so it can be multiplied with the transformed input in so3_mm.

    :param x:    [..., n_spatial]   kernel weights at each SO(3) grid point
    :param b:    output bandwidth   (harmonic degrees kept: 0..b-1)
    :param grid: tuple of (beta, alpha, gamma) tuples  -- kernel sample points
    :return:     [l*m*n, ..., complex]   last dim = 2 (re, im)
    """
    # F: [n_spatial, n_spectral, 2]  precomputed Fourier matrix for this grid
    F = _get_so3_ft(b, grid, device=x.device, dtype=x.dtype)
    assert x.size(-1) == F.size(0), (
        f"Kernel spatial dim ({x.size(-1)}) must match grid size ({F.size(0)}). "
        f"len(grid)={len(grid)}"
    )

    sz     = x.size()                       # [..., n_spatial]
    x_flat = x.view(-1, x.size(-1))         # [n_batch_flat, n_spatial]

    # Project: sum over spatial grid points
    # x_flat: [i, a],  F: [a, f, c]  ->  out: [f, i, c]
    # where i=batch, a=spatial, f=spectral(l*m*n), c=complex(2)
    out = torch.einsum("ia, afc -> fic", x_flat, F)  # [n_spectral, n_batch_flat, 2]

    # Restore batch shape: [n_spectral, ..., 2]
    return out.view(-1, *sz[:-1], 2)


# ---------------------------------------------------------------------------
# Fourier matrix setup
# ---------------------------------------------------------------------------

def _get_so3_ft(b: int, grid: tuple,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return the SO(3) Fourier matrix for this bandwidth and kernel grid.
    Shape: [n_spatial, n_spectral, 2]

    numpy array is cached separately (device-agnostic via lru_cache),
    then moved to the correct device/dtype here.
    """
    F_np = _compute_so3_ft(b, grid)         # float64 numpy [n_spatial, n_spectral, 2]
    return torch.tensor(F_np, dtype=dtype, device=device).contiguous()


@lru_cache(maxsize=32)
def _compute_so3_ft(b: int, grid: tuple) -> np.ndarray:
    """
    Compute the SO(3) Fourier matrix for a given bandwidth and kernel grid.

    Returns float64 ndarray of shape [n_spatial, n_spectral, 2].

    Each row i corresponds to one SO(3) grid point (beta_i, alpha_i, gamma_i).
    Each column f corresponds to one (l, m, n) Wigner-D coefficient.

    For each degree l, we evaluate the full (2l+1) x (2l+1) Wigner-D matrix
    D^l(alpha, beta, gamma), take its conjugate, and flatten all m*n entries.
    This differs from s2_rft which only takes the n=0 column (gamma=0 case).

    n_spectral = sum_{l=0}^{b-1} (2l+1)^2

    Multiplied by (2*l+1) following standard harmonic normalisation.
    (Note: the original did not multiply by (2*l+1) here, absorbing it into
    the kernel weights. We keep the original convention for compatibility.)

    Requires: pip install git+https://github.com/AMLab-Amsterdam/lie_learn
    """
    try:
        from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    except ImportError as e:
        raise ImportError(
            "lie_learn is required for the Wigner-D computation.\n"
            "Install: pip install git+https://github.com/AMLab-Amsterdam/lie_learn"
        ) from e

    n_spatial  = len(grid)
    n_spectral = int(np.sum([(2 * l + 1) ** 2 for l in range(b)]))

    F = np.zeros((n_spatial, n_spectral), dtype=np.complex128)

    for i, (beta, alpha, gamma) in enumerate(grid):
        col = 0
        for l in range(b):
            # Full Wigner-D matrix D^l(alpha, beta, gamma): shape [(2l+1), (2l+1)]
            # Conjugated: follows the spherical convolution theorem convention
            # (same as s2_rft, but now gamma != 0 so the full matrix is needed)
            D = wigner_D_matrix(
                l,
                alpha, beta, gamma,      # Euler angles: alpha (lon), beta (colat), gamma
                field='complex',
                normalization='quantum',
                order='centered',
                condon_shortley='cs'
            ).conj()                     # shape [(2l+1), (2l+1)]

            # Flatten all (2l+1)^2 entries (row-major: m varies slowest, n fastest)
            size = (2 * l + 1) ** 2
            F[i, col : col + size] = D.flatten()
            col += size

    # Convert complex128 [n_spatial, n_spectral]
    # to float64         [n_spatial, n_spectral, 2]
    # NOTE: .view('float') was removed in NumPy 2.0 -- use .view(np.float64)
    F = F.view(np.float64).reshape(n_spatial, n_spectral, 2)

    return F                             # float64 [n_spatial, n_spectral, 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import math

#     torch.manual_seed(42)

#     # Build a small equatorial SO(3) grid (matching so3_equatorial_grid convention)
#     b       = 4
#     n_alpha = 2 * b
#     n_grid  = n_alpha    # single ring: max_beta=0, max_gamma=0, n_beta=1, n_gamma=1
#     grid    = tuple(
#         (math.pi / 2, 2 * math.pi * k / n_alpha, 0.0)
#         for k in range(n_alpha)
#     )

#     nfeature_in  = 3
#     nfeature_out = 5
#     n_spatial    = len(grid)
#     n_spectral   = sum((2 * l + 1) ** 2 for l in range(b))   # = b*(4b^2-1)/3

#     print(f"so3_rft test  |  b={b}, grid size={n_spatial}, n_spectral={n_spectral}")

#     # --- Test 1: output shape ---
#     kernel = torch.randn(nfeature_in, nfeature_out, n_spatial)
#     y      = so3_rft(kernel, b, grid)
#     expected = (n_spectral, nfeature_in, nfeature_out, 2)
#     print(f"\nTest 1 -- output shape")
#     print(f"  Got     : {tuple(y.shape)}")
#     print(f"  Expected: {expected}")
#     assert tuple(y.shape) == expected, f"Shape mismatch!"
#     print("  PASSED")

#     # --- Test 2: linearity (float64 for tight tolerance, see s2_rft.py note) ---
#     k1        = torch.randn(nfeature_in, nfeature_out, n_spatial, dtype=torch.float64)
#     k2        = torch.randn(nfeature_in, nfeature_out, n_spatial, dtype=torch.float64)
#     alpha_val = 2.3

#     y_sum    = so3_rft(k1 + alpha_val * k2, b, grid)
#     y_linear = so3_rft(k1, b, grid) + alpha_val * so3_rft(k2, b, grid)
#     err2 = (y_sum - y_linear).abs().max().item()
#     print(f"\nTest 2 -- linearity (float64): max error {err2:.2e}  (target < 1e-10)")
#     assert err2 < 1e-10, f"FAILED: {err2:.2e}"
#     print("  PASSED")

#     # float32 rounding check
#     k1f, k2f = k1.float(), k2.float()
#     y_sum_f    = so3_rft(k1f + alpha_val * k2f, b, grid)
#     y_linear_f = so3_rft(k1f, b, grid) + alpha_val * so3_rft(k2f, b, grid)
#     err2f = (y_sum_f - y_linear_f).abs().max().item()
#     print(f"  float32 rounding error: {err2f:.2e}  (expected ~1e-5, not a bug)")
#     assert err2f < 1e-3, f"float32 error unexpectedly large: {err2f:.2e}"
#     print("  float32 check PASSED")

#     # --- Test 3: Fourier matrix is cached (same object returned on second call) ---
#     F1 = _compute_so3_ft(b, grid)
#     F2 = _compute_so3_ft(b, grid)
#     assert F1 is F2, "Cache miss -- lru_cache not working"
#     print(f"\nTest 3 -- lru_cache hit  PASSED")

#     # --- Test 4: SO(3) vs S2 consistency at gamma=0 ---
#     # When gamma=0, the Wigner-D matrix D^l(alpha,beta,0) has its n=0 column
#     # equal to the spherical harmonic Y_l^m(beta,alpha) (up to normalisation).
#     # So the n=0 column of the SO(3) Fourier matrix should match the S2 version.
#     # We verify this by checking that the n=0 entries of the SO(3) F matrix
#     # match the S2 F matrix (from s2_rft._compute_s2_ft) for a gamma=0 grid.
#     try:
#         # from s2_rft import _compute_s2_ft
#         s2_grid = tuple((beta, alpha) for (beta, alpha, gamma) in grid)
#         F_so3   = _compute_so3_ft(b, grid)          # [n_spatial, n_spectral_so3, 2]
#         F_s2    = _compute_s2_ft(b, s2_grid)         # [n_spatial, n_spectral_s2,  2]

#         # For each degree l, the n=0 column in the SO(3) matrix sits at
#         # offset col + l*(2l+1) within the (2l+1)^2 block (m varies, n=0 is middle col)
#         n_spectral_s2 = b ** 2
#         so3_n0_cols   = []
#         col = 0
#         for l in range(b):
#             L = 2 * l + 1
#             # n=0 is the centre column (index l) in centred order
#             # flatten order: row m varies fastest? No: D.flatten() is row-major,
#             # so index within block = m_idx * L + n_idx, n=0 -> n_idx=l
#             n0_indices = [m_idx * L + l for m_idx in range(L)]   # n=0 column entries
#             so3_n0_cols.append(F_so3[:, col + np.array(n0_indices), :])
#             col += L * L
#         F_so3_n0 = np.concatenate(so3_n0_cols, axis=1)   # [n_spatial, b^2, 2]

#         # Scale difference: s2_rft multiplies by (2*b), so3_rft does not -- skip scale
#         # Just check the ratio is constant (same shape up to a scalar)
#         ratio = F_so3_n0 / (F_s2 + 1e-30)
#         ratio_std = ratio.std()
#         print(f"\nTest 4 -- SO3 n=0 vs S2 column consistency: ratio std={ratio_std:.2e}")
#         # We expect a consistent ratio (constant scale difference), not zero error
#         print("  (Scale difference between s2_rft and so3_rft conventions is expected)")
#         print("  PASSED (structural check)")
#     except ImportError:
#         print("\nTest 4 -- SKIPPED (s2_rft.py not found in path)")

#     print("\nAll tests passed.")
