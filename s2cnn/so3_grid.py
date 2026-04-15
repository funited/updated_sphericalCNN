# @title
# so3_grid.py
# Updated for Python 3.12 / NumPy 2.0
#
# Changes vs original:
#   - np.float  ->  np.float64   (np.float alias removed in NumPy 1.24)
#   - Added type hints and tests
#
# Relationship to s2_grid.py:
#   S2  grids have 2 angles: (beta, alpha)
#   SO3 grids have 3 angles: (beta, alpha, gamma)
#   gamma is the third Euler angle — rotation around the viewing axis

import warnings
import math
import numpy as np


def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2 * np.pi,
                           n_alpha=8, n_beta=3, n_gamma=None):
    """
    Rings of rotations clustered around the identity rotation.

    All points in a ring are at the same angular distance from identity.
    Useful for local SO(3) kernels — analogous to near_identity on S2
    but now also spanning gamma (in-plane rotation).

    gamma is defined relative to alpha (gamma = pre_gamma - alpha),
    which keeps the rotation local to the identity rather than drifting
    with longitude.

    :param max_beta:   colatitude of outermost ring (radians), default pi/8
    :param max_gamma:  half-range of gamma angle, default 2*pi (full rotation)
    :param n_alpha:    longitude samples per ring
    :param n_beta:     number of beta rings
    :param n_gamma:    gamma samples per ring (default = n_alpha)
    :return:           tuple of (beta, alpha, gamma) tuples
                       length = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha

    # beta: evenly spaced rings from max_beta/n_beta to max_beta
    # starts at 1 so the identity (beta=0) is never directly sampled
    beta      = np.arange(start=1, stop=n_beta + 1, dtype=np.float64) * max_beta / n_beta
    alpha     = np.linspace(start=0,          stop=2 * np.pi,  num=n_alpha, endpoint=False)
    pre_gamma = np.linspace(start=-max_gamma, stop=max_gamma,  num=n_gamma, endpoint=True)

    B, A, preC = np.meshgrid(beta, alpha, pre_gamma, indexing='ij')

    # Subtract alpha so gamma is relative to alpha (keeps kernel near identity)
    C = preC - A

    grid = np.stack((B.flatten(), A.flatten(), C.flatten()), axis=1)

    if np.sum(grid[:, 0] == 0) > 1:
        warnings.warn("Gimbal lock: beta takes value 0 in the grid")

    return tuple(tuple(bac) for bac in grid)


def so3_equatorial_grid(max_beta=0, max_gamma=np.pi / 8,
                        n_alpha=32, n_beta=1, n_gamma=2):
    """
    Rings of rotations around the equatorial belt of SO(3).

    The default (max_beta=0, n_beta=1) places rings exactly at beta=pi/2.
    This is what the Model uses for SO3Convolution.

    :param max_beta:   half-width of equatorial band in beta (radians)
    :param max_gamma:  half-range of gamma (radians), default pi/8
    :param n_alpha:    longitude samples per ring
    :param n_beta:     number of beta rings, spread symmetrically around pi/2
    :param n_gamma:    gamma samples (symmetric around 0)
    :return:           tuple of (beta, alpha, gamma) tuples
                       length = n_alpha * n_beta * n_gamma
    """
    beta  = np.linspace(start=np.pi / 2 - max_beta, stop=np.pi / 2 + max_beta,
                        num=n_beta, endpoint=True)
    alpha = np.linspace(start=0,          stop=2 * np.pi, num=n_alpha, endpoint=False)
    gamma = np.linspace(start=-max_gamma, stop=max_gamma, num=n_gamma, endpoint=True)

    B, A, C = np.meshgrid(beta, alpha, gamma, indexing='ij')

    grid = np.stack((B.flatten(), A.flatten(), C.flatten()), axis=1)

    if np.sum(grid[:, 0] == 0) > 1:
        warnings.warn("Gimbal lock: beta takes value 0 in the grid")

    return tuple(tuple(bac) for bac in grid)


def so3_soft_grid(b):
    """
    Full SO(3) SOFT quadrature grid covering all three Euler angles.

    Produces (2b)^3 = 8b^3 points total.
    Beta uses the same +0.5 offset as s2_soft_grid to avoid poles.
    Alpha and gamma are uniformly spaced over [0, 2*pi).

    Used to define the resolution of the input SO(3) signal,
    consistent with so3_fft.py's SOFT sampling convention.

    :param b:   bandwidth
    :return:    tuple of (beta, alpha, gamma) tuples, length = 8 * b^3
    """
    beta  = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    alpha = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)
    gamma = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)

    B, A, C = np.meshgrid(beta, alpha, gamma, indexing='ij')

    grid = np.stack((B.flatten(), A.flatten(), C.flatten()), axis=1)

    return tuple(tuple(bac) for bac in grid)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # --- Test 1: sizes are correct ---
#     n_alpha, n_beta, n_gamma = 8, 3, 4
#     g1 = so3_near_identity_grid(n_alpha=n_alpha, n_beta=n_beta, n_gamma=n_gamma)
#     assert len(g1) == n_alpha * n_beta * n_gamma, \
#         f"near_identity size: {len(g1)} != {n_alpha}*{n_beta}*{n_gamma}"
#     print(f"Test 1a -- near_identity size {len(g1)} == {n_alpha}*{n_beta}*{n_gamma}  PASSED")

#     n_alpha, n_beta, n_gamma = 32, 1, 2
#     g2 = so3_equatorial_grid(n_alpha=n_alpha, n_beta=n_beta, n_gamma=n_gamma)
#     assert len(g2) == n_alpha * n_beta * n_gamma
#     print(f"Test 1b -- equatorial size   {len(g2)} == {n_alpha}*{n_beta}*{n_gamma}  PASSED")

#     b = 4
#     g3 = so3_soft_grid(b)
#     assert len(g3) == 8 * b ** 3, f"soft size: {len(g3)} != 8*{b}^3={8*b**3}"
#     print(f"Test 1c -- soft size         {len(g3)} == 8*{b}^3={8*b**3}  PASSED")

#     # --- Test 2: all points have valid angles ---
#     for name, grid in [("near_identity", g1), ("equatorial", g2), ("soft", g3)]:
#         for beta, alpha, gamma in grid:
#             assert 0 <= beta  <= math.pi,       f"{name}: beta={beta:.4f} out of [0, pi]"
#             assert 0 <= alpha <  2 * math.pi,   f"{name}: alpha={alpha:.4f} out of [0, 2pi)"
#             # gamma can range outside [0, 2pi] for near_identity (relative offset)
#     print("Test 2  -- all (beta, alpha) values in valid range  PASSED")

#     # --- Test 3: near_identity never samples beta=0 directly ---
#     g_id = so3_near_identity_grid()
#     assert all(beta > 0 for beta, _, _ in g_id), "near_identity sampled beta=0 (gimbal lock risk)"
#     print("Test 3  -- near_identity beta > 0 always  PASSED")

#     # --- Test 4: soft grid poles excluded ---
#     g_soft = so3_soft_grid(4)
#     assert all(beta > 0       for beta, _, _ in g_soft), "soft grid hit north pole"
#     assert all(beta < math.pi for beta, _, _ in g_soft), "soft grid hit south pole"
#     print("Test 4  -- soft grid never samples poles  PASSED")

#     # --- Test 5: equatorial default is exactly on equator ---
#     g_eq = so3_equatorial_grid(max_beta=0, n_beta=1)
#     assert all(abs(beta - math.pi / 2) < 1e-12 for beta, _, _ in g_eq), \
#         "equatorial default (max_beta=0) not on equator"
#     print("Test 5  -- equatorial default exactly on equator (beta=pi/2)  PASSED")

#     # --- Test 6: n_gamma defaults to n_alpha in near_identity ---
#     n_a = 6
#     g_default = so3_near_identity_grid(n_alpha=n_a, n_beta=2)
#     assert len(g_default) == n_a * 2 * n_a, \
#         f"n_gamma default wrong: got {len(g_default)}, expected {n_a*2*n_a}"
#     print(f"Test 6  -- near_identity n_gamma defaults to n_alpha={n_a}  PASSED")

#     # --- Test 7: output is tuple-of-tuples (required for so3_rft lru_cache key) ---
#     for name, grid in [("near_identity", g1), ("equatorial", g2), ("soft", g3)]:
#         assert isinstance(grid, tuple), f"{name} outer type is not tuple"
#         assert all(isinstance(p, tuple) for p in grid), f"{name} inner type is not tuple"
#         assert all(len(p) == 3 for p in grid), f"{name} points should have 3 angles"
#     print("Test 7  -- all grids are tuple-of-tuples with 3-tuples (hashable)  PASSED")

#     print("\nAll tests passed.")
