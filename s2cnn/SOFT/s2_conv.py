# @title
# s2_conv.py
# Updated for Python 3.12 / NumPy 2.0 / PyTorch 2.10
#
# Changes vs original:
#   - Imports updated to use our new modules (s2_fft, so3_fft, s2_rft, s2_mm)
#     instead of the old s2cnn package imports
#   - No mathematical changes -- this is the core of the model and must stay identical

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

from s2_fft import S2_fft_real
from so3_fft import SO3_ifft_real
from s2_rft  import s2_rft
from s2_mm   import s2_mm


class S2Convolution(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        """
        Spherical convolution: lifts a signal from S2 to SO(3).

        Pipeline:
            x  [B, f_in,  2b_in,  2b_in]
              -> S2_fft_real  (forward SHT)          [l*m, B, f_in, 2]
              -> s2_rft       (kernel: grid -> spec)  [l*m, f_in, f_out, 2]
              -> s2_mm        (convolution theorem)   [l*m*n, B, f_out, 2]
              -> SO3_ifft_real (inverse SO3 FFT)      [B, f_out, 2b_out, 2b_out, 2b_out]
              -> + bias

        :param nfeature_in:  number of input  channels
        :param nfeature_out: number of output channels
        :param b_in:         input  bandwidth  (input  grid is 2b_in  x 2b_in)
        :param b_out:        output bandwidth  (output grid is 2b_out x 2b_out x 2b_out)
        :param grid:         tuple of (beta, alpha) tuples -- kernel sample points on S2
        """
        super(S2Convolution, self).__init__()
        self.nfeature_in  = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in         = b_in
        self.b_out        = b_out
        self.grid         = grid

        # Learnable kernel: one scalar weight per (f_in, f_out, grid_point)
        self.kernel = Parameter(
            torch.empty(nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1)
        )

        # Fixed scaling -- keeps activations well-conditioned regardless of
        # bandwidth or feature count. Applied outside the parameters so ADAM
        # sees order-1 gradients. Compensates for:
        #   len(grid)      : sum over kernel grid points in s2_rft
        #   nfeature_in    : sum over input channels in s2_mm
        #   b_out^4/b_in^2 : spectral energy scaling from bandwidth change (S2 is 2D)
        self.scaling = 1.0 / math.sqrt(
            len(self.grid) * self.nfeature_in * (self.b_out ** 4.0) / (self.b_in ** 2.0)
        )

        # Learnable bias: one scalar per output channel, broadcast over SO(3) volume
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1))

    def forward(self, x):
        """
        :param x:    [batch, feature_in,  2*b_in,  2*b_in]   signal on S2
        :return:     [batch, feature_out, 2*b_out, 2*b_out, 2*b_out]   signal on SO3
        """
        assert x.size(1) == self.nfeature_in, \
            f"Expected {self.nfeature_in} input channels, got {x.size(1)}"
        assert x.size(2) == 2 * self.b_in, \
            f"Expected spatial dim {2*self.b_in} (=2*b_in), got {x.size(2)}"
        assert x.size(3) == 2 * self.b_in, \
            f"Expected spatial dim {2*self.b_in} (=2*b_in), got {x.size(3)}"

        # Step 1: forward SHT -- signal from S2 spatial to spectral
        x = S2_fft_real.apply(x, self.b_out)
        # x: [l*m, batch, f_in, 2]

        # Step 2: kernel from grid samples to spectral
        y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)
        # y: [l*m, f_in, f_out, 2]

        # Step 3: spectral multiplication -- convolution theorem on S2
        # creates the extra n-index, lifting the output from S2 to SO(3)
        z = s2_mm(x, y)
        # z: [l*m*n, batch, f_out, 2]

        # Step 4: inverse SO(3) FFT -- spectral back to SO(3) spatial
        z = SO3_ifft_real.apply(z)
        # z: [batch, f_out, 2*b_out, 2*b_out, 2*b_out]

        return z + self.bias


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # from s2_grid import s2_equatorial_grid
#     # from s2_fft  import s2_ifft

#     torch.manual_seed(42)

#     nfeature_in  = 6
#     nfeature_out = 16
#     b_in         = 8
#     b_out        = 4
#     grid         = s2_equatorial_grid(max_beta=0, n_alpha=2 * b_in, n_beta=1)

#     print(f"S2Convolution test")
#     print(f"  f_in={nfeature_in}, f_out={nfeature_out}, b_in={b_in}, b_out={b_out}")
#     print(f"  kernel grid size : {len(grid)}")
#     print(f"  input  shape     : [B, {nfeature_in}, {2*b_in}, {2*b_in}]")
#     print(f"  output shape     : [B, {nfeature_out}, {2*b_out}, {2*b_out}, {2*b_out}]")

#     conv = S2Convolution(nfeature_in, nfeature_out, b_in, b_out, grid)

#     # Build a bandlimited input signal via iSHT of random spectral coefficients
#     # (plain torch.randn in spatial domain is not bandlimited -- see s2_fft tests)
#     nsp_s2    = b_in ** 2
#     spec_in   = torch.randn(nsp_s2, 2, nfeature_in, 2)    # [l*m, B, f_in, 2]
#     x_spatial = s2_ifft(spec_in, b_out=b_in)[..., 0]      # [2, f_in, 2b, 2b]

#     # --- Test 1: output shape ---
#     y = conv(x_spatial)
#     expected = (2, nfeature_out, 2 * b_out, 2 * b_out, 2 * b_out)
#     assert tuple(y.shape) == expected, f"Shape mismatch: {tuple(y.shape)} != {expected}"
#     print(f"\nTest 1 -- output shape {tuple(y.shape)}  PASSED")

#     # --- Test 2: output is finite ---
#     assert torch.isfinite(y).all(), "Output contains NaN or Inf"
#     print(f"Test 2 -- output is finite  PASSED")

#     # --- Test 3: gradients flow to kernel and bias ---
#     y.sum().backward()
#     assert conv.kernel.grad is not None, "No gradient for kernel"
#     assert conv.bias.grad   is not None, "No gradient for bias"
#     print(f"Test 3 -- gradients flow to kernel and bias  PASSED")

#     # --- Test 4: bias is applied correctly ---
#     conv2 = S2Convolution(nfeature_in, nfeature_out, b_in, b_out, grid)
#     with torch.no_grad():
#         conv2.kernel.copy_(conv.kernel)
#         conv2.bias.zero_()
#     y_no_bias = conv2(x_spatial)
#     delta     = (y - y_no_bias - conv.bias).abs().max().item()
#     print(f"Test 4 -- bias applied correctly:  max error {delta:.2e}  (target < 1e-5)")
#     assert delta < 1e-5, f"FAILED: {delta:.2e}"
#     print("  PASSED")

#     # --- Test 5: scaling is < 1 (kernel is attenuated as expected) ---
#     print(f"Test 5 -- scaling factor = {conv.scaling:.4e}  (should be << 1)")
#     assert conv.scaling < 1.0
#     print("  PASSED")

#     # --- Test 6: linearity in x ---
#     spec_in2  = torch.randn(nsp_s2, 2, nfeature_in, 2)
#     x2        = s2_ifft(spec_in2, b_out=b_in)[..., 0]
#     alpha     = 1.7
#     y_sum     = conv(x_spatial + alpha * x2)
#     y_linear  = conv(x_spatial) + alpha * conv(x2)
#     err6      = (y_sum - y_linear).abs().max().item()
#     print(f"Test 6 -- linearity in x:  max error {err6:.2e}  (target < 1e-4)")
#     assert err6 < 1e-4, f"FAILED: {err6:.2e}"
#     print("  PASSED")

#     print("\nAll tests passed.")
