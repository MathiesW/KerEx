from keras import KerasTensor
from keras import ops
from keras import saving
from typing import Tuple, List
from keras.src.backend.config import backend
from .base_spectral_conv import BaseSpectralConv1D


if backend() == "tensorflow":
    @saving.register_keras_serializable(package="KerasAddon.Layers.FNO", name="SpectralConv1D")
    class SpectralConv1D(BaseSpectralConv1D):
        def call(self, x: KerasTensor) -> Tuple[KerasTensor, callable]:
            @ops.custom_gradient
            def forward(x: KerasTensor) -> Tuple[KerasTensor, callable]:
                x_hat_real, x_hat_imag = self.rfft(x)  # shape = (None, x//2+1, ch_out)

                # reduce to relevant modes
                x_hat_real_truncated = x_hat_real[self.mode_truncation_slice]  # shape = (None, m, ch_out)
                x_hat_imag_truncated = x_hat_imag[self.mode_truncation_slice]  # shape = (None, m, ch_out)

                y_hat_real_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_truncated, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_truncated, self.imag_weights)
                y_hat_imag_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_truncated, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_truncated, self.imag_weights)

                y_hat_real = ops.pad(y_hat_real_truncated, pad_width=self.pad_width)
                y_hat_imag = ops.pad(y_hat_imag_truncated, pad_width=self.pad_width)

                # add bias, shape = (None, m, ch_out)
                if self.use_bias:
                    y_hat_real += self.bias
                    y_hat_imag += self.bias

                # reconstruct y via irfft, shape = (None, x, ch_out)
                y = self.irfft((y_hat_real, y_hat_imag))

                def backprop(dy: KerasTensor, variables=None) -> Tuple[KerasTensor, List[KerasTensor]]:
                    # get real and imaginary part via rfft, shape = (None, x//2+1, ch_out)
                    dy_hat_real, dy_hat_imag = self.rfft(dy)

                    # reduce to relevant modes, shape = (None, m, ch_out)
                    dy_hat_real_truncated = dy_hat_real[self.mode_truncation_slice]
                    dy_hat_imag_truncated = dy_hat_imag[self.mode_truncation_slice]

                    # compute gradients for weights, shape = (ch_in, m, ch_out)
                    dw_real = ops.einsum(self.einsum_op_backprop_weights, dy_hat_real_truncated, x_hat_real_truncated) + ops.einsum(self.einsum_op_backprop_weights, dy_hat_imag_truncated, x_hat_imag_truncated)
                    dw_imag = ops.einsum(self.einsum_op_backprop_weights, dy_hat_real_truncated, x_hat_imag_truncated) - ops.einsum(self.einsum_op_backprop_weights, dy_hat_imag_truncated, x_hat_real_truncated)

                    if self.use_bias:
                        # compute gradients for bias, shape = (ch_out, )
                        db = ops.einsum(self.einsum_op_backprop_bias, dy_hat_real_truncated + dy_hat_imag_truncated)

                    # compute gradient for inputs, shape = (None, m, ch_in)
                    dx_hat_real_truncated = ops.einsum(self.einsum_op_backprop_x, dy_hat_real_truncated, self.real_weights) + ops.einsum(self.einsum_op_backprop_x, dy_hat_imag_truncated, self.imag_weights)
                    dx_hat_imag_truncated = ops.einsum(self.einsum_op_backprop_x, dy_hat_imag_truncated, self.real_weights) - ops.einsum(self.einsum_op_backprop_x, dy_hat_real_truncated, self.imag_weights)

                    # pad for ifft, shape = (None, x, ch_in)
                    dx_hat_real = ops.pad(dx_hat_real_truncated, pad_width=self.pad_width)
                    dx_hat_imag = ops.pad(dx_hat_imag_truncated, pad_width=self.pad_width)

                    # apply irfft, shape = (None, x, ch_in)
                    dx = self.irfft((dx_hat_real, dx_hat_imag))
                    if self.use_bias:
                        return dx, [db, dw_real, dw_imag]
                    
                    return dx, [dw_real, dw_imag]

                return y, backprop
                
            return forward(x)

elif backend() == "jax":
    @saving.register_keras_serializable(package="KerasAddon.Layers.FNO", name="SpectralConv1D")
    class SpectralConv1D(BaseSpectralConv1D):
        def call(self, x: KerasTensor) -> KerasTensor:
            # forward pass, shape = (None, x, y, ch_in)
            x_hat_real, x_hat_imag = self.rfft(x)

            # reduce to relevant modes, shape = (None, mx, my, ch_in)
            x_hat_real_reduced = x_hat_real[self.mode_truncation_slice]  # 1D: (batch, m, ch_out), 2D: (batch, mx, my, ch_in)
            x_hat_imag_reduced = x_hat_imag[self.mode_truncation_slice]  # 1D: (batch, m, ch_out), 2D: (batch, mx, my, ch_in)

            y_hat_real_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_reduced, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_reduced, self.imag_weights)
            y_hat_imag_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_reduced, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_reduced, self.imag_weights)

            y_hat_real = ops.pad(y_hat_real_truncated, pad_width=self.pad_width)
            y_hat_imag = ops.pad(y_hat_imag_truncated, pad_width=self.pad_width)

            # add bias, shape = (None, mx, my, ch_out)
            if self.use_bias:
                y_hat_real += self.bias
                y_hat_imag += self.bias

            # reconstruct y via irfft, shape = (None, x, y, ch_out)
            y = self.irfft((y_hat_real, y_hat_imag))

            return y
  
else:
    raise RuntimeError(f"Spectral Convolution is only defined for keras backends 'tensorflow' and 'jax', received {backend()}")