# @title
# so3_conv.py
# Updated for Python 3.12 / NumPy 2.0 / PyTorch 2.10
#
# Changes vs original:
#   - Imports updated to use our new modules (so3_fft, so3_rft, so3_mm)
#     instead of the old s2cnn package imports
#   - No mathematical changes -- stays identical to original

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

# from so3_fft import SO3_fft_real, SO3_ifft_real
# from so3_rft import so3_rft
# from so3_mm  import so3_mm


class SO3Convolution(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        """
        SO(3) convolution: stays within SO(3) (unlike S2Convolution which lifts S2 -> SO3).

        Pipeline:
            x  [B, f_in,  2b_in,  2b_in,  2b_in]
              -> SO3_fft_real  (forward SO3 FFT)       [l*m*n, B, f_in,  2]
              -> so3_rft       (kernel: grid -> spec)   [l*m*n, f_in, f_out, 2]
              -> so3_mm        (spectral multiply)      [l*m*n, B, f_out, 2]
              -> SO3_ifft_real (inverse SO3 FFT)        [B, f_out, 2b_out, 2b_out, 2b_out]
              -> + bias

        Key difference from S2Convolution:
            - Input  is already on SO(3) (3 spatial dims, not 2)
            - Output stays on SO(3) (same nspec in, same nspec out)
            - Kernel grid has 3 angles (beta, alpha, gamma) not 2
            - Scaling uses b_out^3/b_in^3 (SO3 is 3D) not b_out^4/b_in^2

        :param nfeature_in:  number of input  channels
        :param nfeature_out: number of output channels
        :param b_in:         input  bandwidth  (input  grid is 2b_in  x 2b_in  x 2b_in)
        :param b_out:        output bandwidth  (output grid is 2b_out x 2b_out x 2b_out)
        :param grid:         tuple of (beta, alpha, gamma) tuples -- kernel sample points on SO3
        """
        super(SO3Convolution, self).__init__()
        self.nfeature_in  = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in         = b_in
        self.b_out        = b_out
        self.grid         = grid

        # Learnable kernel: one scalar weight per (f_in, f_out, grid_point)
        self.kernel = Parameter(
            torch.empty(nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1)
        )

        # Learnable bias: one scalar per output channel, broadcast over SO(3) volume
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1))

        # Fixed scaling to keep activations well-conditioned for ADAM.
        # Applied outside the parameters so ADAM sees order-1 gradients.
        # Compensates for:
        #   len(grid)      : sum over kernel grid points in so3_rft
        #   nfeature_in    : sum over input channels in so3_mm
        #   b_out^3/b_in^3 : spectral energy scaling (SO3 is 3D, hence ^3 not ^4/^2)
        self.scaling = 1.0 / math.sqrt(
            len(self.grid) * self.nfeature_in * (self.b_out ** 3.0) / (self.b_in ** 3.0)
        )

    def forward(self, x):
        """
        :param x:    [batch, feature_in,  2*b_in,  2*b_in,  2*b_in]   signal on SO3
        :return:     [batch, feature_out, 2*b_out, 2*b_out, 2*b_out]   signal on SO3
        """
        assert x.size(1) == self.nfeature_in, \
            f"Expected {self.nfeature_in} input channels, got {x.size(1)}"
        assert x.size(2) == 2 * self.b_in, \
            f"Expected spatial dim {2*self.b_in} (=2*b_in), got {x.size(2)}"
        assert x.size(3) == 2 * self.b_in, \
            f"Expected spatial dim {2*self.b_in}, got {x.size(3)}"
        assert x.size(4) == 2 * self.b_in, \
            f"Expected spatial dim {2*self.b_in}, got {x.size(4)}"

        # Step 1: forward SO3 FFT -- signal from SO3 spatial to spectral
        x = SO3_fft_real.apply(x, self.b_out)
        # x: [l*m*n, batch, f_in, 2]

        # Step 2: kernel from grid samples to spectral
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)
        # y: [l*m*n, f_in, f_out, 2]

        assert x.size(0) == y.size(0), \
            f"Spectral size mismatch: x={x.size(0)}, y={y.size(0)}"
        assert x.size(2) == y.size(1), \
            f"feature_in mismatch: x={x.size(2)}, y={y.size(1)}"

        # Step 3: spectral multiply -- stays in SO3 (same nspec in and out)
        z = so3_mm(x, y)
        # z: [l*m*n, batch, f_out, 2]

        assert z.size(0) == x.size(0)
        assert z.size(1) == x.size(1)
        assert z.size(2) == y.size(2)

        # Step 4: inverse SO3 FFT -- spectral back to SO3 spatial
        z = SO3_ifft_real.apply(z)
        # z: [batch, f_out, 2*b_out, 2*b_out, 2*b_out]

        return z + self.bias


class SO3Shortcut(Module):
    """
    Skip connection for ResNet-style architectures on SO(3).

    If feature count or bandwidth changes, applies a 1-point SO3Convolution
    (grid = single identity point) to project to the new shape.
    If both are unchanged, passes x through directly.
    """
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out):
        super(SO3Shortcut, self).__init__()
        assert b_out <= b_in, f"SO3Shortcut requires b_out <= b_in, got {b_out} > {b_in}"
        if (nfeature_in != nfeature_out) or (b_in != b_out):
            # Single-point kernel at identity rotation (0,0,0) acts as a
            # 1x1x1 linear projection across channels with bandwidth downsampling
            self.conv = SO3Convolution(
                nfeature_in=nfeature_in, nfeature_out=nfeature_out,
                b_in=b_in, b_out=b_out,
                grid=((0, 0, 0),)
            )
        else:
            self.conv = None

    def forward(self, x):
        """
        :param x:    [batch, feature_in,  2*b_in,  2*b_in,  2*b_in]
        :return:     [batch, feature_out, 2*b_out, 2*b_out, 2*b_out]
        """
        if self.conv is not None:
            return self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # from so3_grid import so3_equatorial_grid
#     # from so3_fft  import so3_rifft, _nspec

#     torch.manual_seed(42)

#     # Scaled-down model parameters (same ratios as original: b_out/b_in = 10/16 ~ 0.625)
#     nfeature_in  = 16
#     nfeature_out = 16
#     b_in         = 4
#     b_out        = 3
#     B            = 2

#     grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2*b_in, n_beta=1, n_gamma=1)

#     conv = SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid)

#     print("SO3Convolution smoke test")
#     print(f"  f_in={nfeature_in}, f_out={nfeature_out}, b_in={b_in}, b_out={b_out}")
#     print(f"  kernel grid size : {len(grid)}")
#     print(f"  kernel shape     : {list(conv.kernel.shape)}")
#     print(f"  scaling          : {conv.scaling:.4e}")
#     print(f"  input  shape     : [B, {nfeature_in}, {2*b_in}, {2*b_in}, {2*b_in}]")
#     print(f"  output shape     : [B, {nfeature_out}, {2*b_out}, {2*b_out}, {2*b_out}]")

#     # Build bandlimited SO3 input via so3_rifft of random spectral coefficients
#     nsp = _nspec(b_in)
#     spec_in = torch.randn(nsp, B, nfeature_in, 2)
#     x       = so3_rifft(spec_in, b_out=b_in)         # [B, f_in, 2b, 2b, 2b]

#     # --- Test 1: output shape ---
#     y = conv(x)
#     expected = (B, nfeature_out, 2*b_out, 2*b_out, 2*b_out)
#     assert tuple(y.shape) == expected, f"Shape mismatch: {tuple(y.shape)}"
#     print(f"\nTest 1 -- output shape {tuple(y.shape)}  PASSED")

#     # --- Test 2: output is finite ---
#     assert torch.isfinite(y).all()
#     print("Test 2 -- output is finite  PASSED")

#     # --- Test 3: gradients flow ---
#     y.sum().backward()
#     assert conv.kernel.grad is not None
#     assert conv.bias.grad   is not None
#     print("Test 3 -- gradients flow  PASSED")

#     # --- Test 4: bias applied correctly ---
#     conv2 = SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid)
#     with torch.no_grad():
#         conv2.kernel.copy_(conv.kernel)
#         conv2.bias.zero_()
#     delta = (y - conv2(x) - conv.bias).abs().max().item()
#     print(f"Test 4 -- bias applied correctly:  max error {delta:.2e}  (target < 1e-5)")
#     assert delta < 1e-5, f"FAILED: {delta:.2e}"
#     print("  PASSED")

#     # --- Test 5: linearity in x ---
#     spec_in2 = torch.randn(nsp, B, nfeature_in, 2)
#     x2       = so3_rifft(spec_in2, b_out=b_in)
#     alpha    = 1.7
#     y_sum    = conv(x + alpha * x2)
#     y_lin    = conv(x) + alpha * conv(x2)
#     err5     = (y_sum - y_lin).abs().max().item()
#     print(f"Test 5 -- linearity in x:  max error {err5:.2e}  (target < 1e-4)")
#     assert err5 < 1e-4, f"FAILED: {err5:.2e}"
#     print("  PASSED")

#     # --- Test 6: SO3Shortcut identity (same features and bandwidth) ---
#     shortcut_id = SO3Shortcut(nfeature_in, nfeature_in, b_in, b_in)
#     assert shortcut_id.conv is None, "Identity shortcut should have conv=None"
#     assert torch.equal(shortcut_id(x), x), "Identity shortcut should return x unchanged"
#     print("Test 6 -- SO3Shortcut identity pass-through  PASSED")

#     # --- Test 7: SO3Shortcut projection (different features or bandwidth) ---
#     shortcut_proj = SO3Shortcut(nfeature_in, nfeature_out * 2, b_in, b_out)
#     assert shortcut_proj.conv is not None, "Projection shortcut should have a conv"
#     y_sc = shortcut_proj(x)
#     assert tuple(y_sc.shape) == (B, nfeature_out * 2, 2*b_out, 2*b_out, 2*b_out)
#     print(f"Test 7 -- SO3Shortcut projection shape {tuple(y_sc.shape)}  PASSED")

#     print("\nAll tests passed.")
