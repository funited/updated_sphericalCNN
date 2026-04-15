# @title
# s2_grid.py
# Updated for Python 3.12 / NumPy 2.0
#
# Changes vs original:
#   - np.float  ->  np.float64   (np.float alias removed in NumPy 1.24)
#   - Added type hints and tests

import numpy as np


def s2_near_identity_grid(max_beta=np.pi / 8, n_alpha=8, n_beta=3):
    """
    Sample points arranged in rings around the north pole.

    Useful for local kernels: the filter only "sees" a small cap
    near the pole, which after spherical convolution can be placed
    anywhere on the sphere.

    :param max_beta:  colatitude of the outermost ring (radians)
                      default pi/8 ~ 22.5 deg from north pole
    :param n_alpha:   number of points per ring (longitude samples)
    :param n_beta:    number of rings
    :return:          tuple of (beta, alpha) tuples, length = n_beta * n_alpha
    """
    # Rings at beta = max_beta/n_beta, 2*max_beta/n_beta, ..., max_beta
    # Note: starts at 1 (not 0) so the pole itself is never sampled
    beta  = np.arange(start=1, stop=n_beta + 1, dtype=np.float64) * max_beta / n_beta
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    B, A  = np.meshgrid(beta, alpha, indexing='ij')
    grid  = np.stack((B.flatten(), A.flatten()), axis=1)
    return tuple(tuple(ba) for ba in grid)


def s2_equatorial_grid(max_beta=0, n_alpha=32, n_beta=1):
    """
    Sample points arranged in rings around the equator.

    The default (max_beta=0, n_beta=1) places a single ring exactly
    on the equator -- a natural choice for rotationally symmetric kernels.
    This is what the S2Convolution in the model uses.

    :param max_beta:  half-width of the equatorial band (radians)
                      0 = single equatorial ring
    :param n_alpha:   number of points per ring
    :param n_beta:    number of rings (spread symmetrically above/below equator)
    :return:          tuple of (beta, alpha) tuples, length = n_beta * n_alpha
    """
    # Rings centred at pi/2 (equator), spread +/- max_beta
    # endpoint=True: both boundary rings are included
    beta  = np.linspace(start=np.pi / 2 - max_beta,
                        stop=np.pi  / 2 + max_beta,
                        num=n_beta, endpoint=True)
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    B, A  = np.meshgrid(beta, alpha, indexing='ij')
    grid  = np.stack((B.flatten(), A.flatten()), axis=1)
    return tuple(tuple(ba) for ba in grid)


def s2_soft_grid(b):
    """
    Full-sphere SOFT (Sampling On Full group using Trapezoids) quadrature grid.

    Covers the entire sphere with 2b x 2b = 4b^2 points.
    The +0.5 offset keeps points strictly away from both poles,
    matching the SOFT sampling theorem for bandwidth-b signals.

    Primarily used to define the spatial resolution of the *input signal*
    (not the kernel), consistent with s2_fft.py's SOFT beta grid.

    :param b:   bandwidth
    :return:    tuple of (beta, alpha) tuples, length = 4 * b^2
    """
    beta  = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    alpha = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)
    B, A  = np.meshgrid(beta, alpha, indexing='ij')
    grid  = np.stack((B.flatten(), A.flatten()), axis=1)
    return tuple(tuple(ba) for ba in grid)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import math

#     # --- Test 1: sizes are correct ---
#     n_alpha, n_beta = 8, 3
#     g1 = s2_near_identity_grid(n_alpha=n_alpha, n_beta=n_beta)
#     assert len(g1) == n_alpha * n_beta, f"near_identity size: {len(g1)} != {n_alpha*n_beta}"
#     print(f"Test 1a -- near_identity size {len(g1)} == {n_alpha}*{n_beta}  PASSED")

#     n_alpha, n_beta = 32, 1
#     g2 = s2_equatorial_grid(n_alpha=n_alpha, n_beta=n_beta)
#     assert len(g2) == n_alpha * n_beta
#     print(f"Test 1b -- equatorial size   {len(g2)} == {n_alpha}*{n_beta}  PASSED")

#     b = 8
#     g3 = s2_soft_grid(b)
#     assert len(g3) == 4 * b ** 2, f"soft size: {len(g3)} != {4*b**2}"
#     print(f"Test 1c -- soft size         {len(g3)} == 4*{b}^2={4*b**2}  PASSED")

#     # --- Test 2: all points are valid (beta in [0,pi], alpha in [0,2pi)) ---
#     for name, grid in [("near_identity", g1), ("equatorial", g2), ("soft", g3)]:
#         for beta, alpha in grid:
#             assert 0 <= beta  <= math.pi,     f"{name}: beta={beta:.4f} out of [0,pi]"
#             assert 0 <= alpha <  2 * math.pi, f"{name}: alpha={alpha:.4f} out of [0,2pi)"
#     print("Test 2  -- all (beta, alpha) values in valid range  PASSED")

#     # --- Test 3: near_identity never samples the pole (beta > 0) ---
#     g_pole = s2_near_identity_grid()
#     assert all(beta > 0 for beta, _ in g_pole), "near_identity sampled the pole!"
#     print("Test 3  -- near_identity never samples pole (beta > 0)  PASSED")

#     # --- Test 4: soft grid poles are excluded (beta != 0 and beta != pi) ---
#     g_soft = s2_soft_grid(4)
#     assert all(beta > 0        for beta, _ in g_soft), "soft grid hit north pole"
#     assert all(beta < math.pi  for beta, _ in g_soft), "soft grid hit south pole"
#     print("Test 4  -- soft grid never samples poles  PASSED")

#     # --- Test 5: equatorial grid default is exactly on equator ---
#     g_eq = s2_equatorial_grid(max_beta=0, n_beta=1)
#     assert all(abs(beta - math.pi / 2) < 1e-12 for beta, _ in g_eq), \
#         "equatorial grid (max_beta=0) not on equator"
#     print("Test 5  -- equatorial default is exactly on equator (beta=pi/2)  PASSED")

#     # --- Test 6: output is a tuple of tuples (required by s2_rft lru_cache) ---
#     for name, grid in [("near_identity", g1), ("equatorial", g2), ("soft", g3)]:
#         assert isinstance(grid, tuple), f"{name} outer type is not tuple"
#         assert all(isinstance(p, tuple) for p in grid), f"{name} inner type is not tuple"
#     print("Test 6  -- all grids are tuple-of-tuples (hashable for lru_cache)  PASSED")

#     print("\nAll tests passed.")
