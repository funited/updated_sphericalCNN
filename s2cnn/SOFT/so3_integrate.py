# @title
# so3_integrate.py
# Updated for Python 3.12 / PyTorch 2.10
#
# Changes vs original:
#   - show_running decorator     -> removed (s2cnn.utils.decorator no longer needed)
#   - device_type/device_index   -> removed from lru_cache key (same pattern as all other files)
#                                   numpy array cached separately, moved to device in _get_weights

from functools import lru_cache
import numpy as np
import torch


def so3_integrate(x):
    """
    Integrate a signal on SO(3) using the Haar measure.

    Sums over alpha and gamma uniformly (they are already uniformly sampled),
    then does a weighted sum over beta using SOFT quadrature weights that
    account for the sin(beta) factor in the spherical measure.

    :param x:   [..., beta, alpha, gamma]   shape (..., 2b, 2b, 2b)
    :return:    [...]   one scalar per batch/channel element
    """
    assert x.size(-1) == x.size(-2) == x.size(-3), \
        f"Expected equal spatial dims, got {x.size(-3)}, {x.size(-2)}, {x.size(-1)}"

    b = x.size(-1) // 2
    w = _get_weights(b, device=x.device, dtype=x.dtype)   # [2*b]

    # Sum uniformly over gamma and alpha (both uniformly sampled)
    x = x.sum(dim=-1)    # [..., beta, alpha]
    x = x.sum(dim=-1)    # [..., beta]

    # Weighted sum over beta using SOFT quadrature weights
    # w accounts for sin(beta) * d_beta in the Haar measure
    x = (x * w).sum(dim=-1)   # [...]

    return x


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _get_weights(b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return SOFT quadrature weights for bandwidth b as a Tensor on `device`.
    Shape: [2*b]

    numpy array cached separately (device-agnostic),
    moved to device/dtype here without polluting the lru_cache key.
    """
    w = _compute_weights(b)    # float64 numpy [2*b]
    return torch.tensor(w, dtype=dtype, device=device)


@lru_cache(maxsize=32)
def _compute_weights(b: int) -> np.ndarray:
    """
    Compute SOFT quadrature weights for bandwidth b.
    Returns float64 ndarray of shape [2*b].

    These weights w_i satisfy:
        sum_i  w_i * f(beta_i)  ~  integral_0^pi  f(beta) sin(beta) d_beta

    for any function f bandlimited to degree b-1.

    Requires: pip install git+https://github.com/AMLab-Amsterdam/lie_learn
    """
    try:
        import lie_learn.spaces.S3 as S3
    except ImportError as e:
        raise ImportError(
            "lie_learn is required.\n"
            "Install: pip install git+https://github.com/AMLab-Amsterdam/lie_learn"
        ) from e

    return np.array(S3.quadrature_weights(b), dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import math
#     torch.manual_seed(42)

#     b = 8

#     # --- Test 1: output shape ---
#     x = torch.randn(3, 5, 2 * b, 2 * b, 2 * b)   # [batch=3, channels=5, beta, alpha, gamma]
#     y = so3_integrate(x)
#     assert tuple(y.shape) == (3, 5), f"Shape mismatch: {tuple(y.shape)}"
#     print(f"Test 1 -- output shape {tuple(y.shape)} == (3, 5)  PASSED")

#     # --- Test 2: linearity ---
#     x1 = torch.randn_like(x)
#     x2 = torch.randn_like(x)
#     a  = 3.7
#     err2 = (so3_integrate(x1 + a * x2) - (so3_integrate(x1) + a * so3_integrate(x2))).abs().max().item()
#     print(f"Test 2 -- linearity:  max error {err2:.2e}  (target < 1e-5)")
#     assert err2 < 1e-5, f"FAILED: {err2:.2e}"
#     print("  PASSED")

#     # --- Test 3: constant signal integrates to 1 ---
#     # lie_learn's S3.quadrature_weights are normalised so that the Haar
#     # measure on SO(3) has total mass 1 (probability measure convention).
#     # With uniform alpha and gamma sums each contributing a factor of 2b,
#     # and weights summing to 1/(4b^2), the total is:
#     #   2b  (alpha sum)  x  2b  (gamma sum)  x  1/(4b^2)  (beta weights)  =  1
#     # So integrate(ones) == 1.0 exactly.
#     ones = torch.ones(1, 2 * b, 2 * b, 2 * b)
#     got  = so3_integrate(ones).item()
#     err3 = abs(got - 1.0)
#     print(f"Test 3 -- integrate(ones) = {got:.6f},  expected 1.0  (abs err {err3:.2e})")
#     assert err3 < 1e-5, f"FAILED: abs error {err3:.2e}"
#     print("  PASSED")

#     # --- Test 4: gradients flow ---
#     x_g = x.clone().requires_grad_(True)
#     so3_integrate(x_g).sum().backward()
#     assert x_g.grad is not None, "Gradient did not flow"
#     print(f"Test 4 -- gradients flow  PASSED")

#     # --- Test 5: GPU == CPU (if CUDA available) ---
#     if torch.cuda.is_available():
#         y_cpu = so3_integrate(x)
#         y_gpu = so3_integrate(x.cuda()).cpu()
#         rel   = (y_cpu - y_gpu).abs().max().item() / (y_cpu.std().item() + 1e-8)
#         print(f"Test 5 -- GPU vs CPU:  relative error {rel:.2e}  (target < 1e-5)")
#         assert rel < 1e-5, f"FAILED: {rel:.2e}"
#         print("  PASSED")
#     else:
#         print("Test 5 -- GPU vs CPU:  SKIPPED (no CUDA device)")

#     print("\nAll tests passed.")
