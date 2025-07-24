from keras import KerasTensor
from keras import ops
from keras import saving
from typing import Tuple, List
from keras.src.backend.config import backend
from .base_spectral_conv import BaseSpectralConv2D


if backend() == "tensorflow":
    @saving.register_keras_serializable(package="KerasAddon.Layers.FNO", name="SpectralConv2D")
    class SpectralConv2D(BaseSpectralConv2D):
        def call(self, x: KerasTensor) -> Tuple[KerasTensor, callable]:
            @ops.custom_gradient
            def forward(x: KerasTensor) -> Tuple[KerasTensor, callable]:
                x_hat_real, x_hat_imag = self.rfft(x)

                x_hat_real_truncated = ops.concatenate([x_hat_real[self.mode_truncation_slice_pos], x_hat_real[self.mode_truncation_slice_neg]], axis=self.data_axes[0])
                x_hat_imag_truncated = ops.concatenate([x_hat_imag[self.mode_truncation_slice_pos], x_hat_imag[self.mode_truncation_slice_neg]], axis=self.data_axes[0])

                # apply weights to spectral convolution, shape = (None, my, mx, ch_out)
                y_hat_real_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_truncated, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_truncated, self.imag_weights)
                y_hat_imag_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_truncated, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_truncated, self.imag_weights)

                # now we have (None, my // 2, mx, ch_out)
                y_hat_real_truncated_pos, y_hat_real_truncated_neg = ops.split(y_hat_real_truncated, indices_or_sections=2, axis=self.data_axes[0])
                y_hat_imag_truncated_pos, y_hat_imag_truncated_neg = ops.split(y_hat_imag_truncated, indices_or_sections=2, axis=self.data_axes[0])

                # now we have (None, y // 2, x, ch_out)
                y_hat_real_pos = ops.pad(y_hat_real_truncated_pos, pad_width=self.pad_width_pos)
                y_hat_real_neg = ops.pad(y_hat_real_truncated_neg, pad_width=self.pad_width_neg)
                y_hat_imag_pos = ops.pad(y_hat_imag_truncated_pos, pad_width=self.pad_width_pos)
                y_hat_imag_neg = ops.pad(y_hat_imag_truncated_neg, pad_width=self.pad_width_neg)

                y_hat_real = ops.concatenate([y_hat_real_pos, y_hat_real_neg], axis=self.data_axes[0])
                y_hat_imag = ops.concatenate([y_hat_imag_pos, y_hat_imag_neg], axis=self.data_axes[0])    

                # add bias, shape = (None, mx, my, ch_out)
                if self.use_bias:
                    y_hat_real += self.bias
                    y_hat_imag += self.bias

                # reconstruct y via irfft, shape = (None, x, y, ch_out)
                y = self.irfft((y_hat_real, y_hat_imag))

                def backprop(dy: KerasTensor, variables=None) -> Tuple[KerasTensor, List[KerasTensor]]:
                    # get real and imaginary part via rfft, shape = (None, x, y//2+1, ch_out)
                    dy_hat_real, dy_hat_imag = self.rfft(dy)

                    # reduce to relevant modes, shape = (None, mx, my, ch_out)
                    dy_hat_real_truncated = ops.concatenate([dy_hat_real[self.mode_truncation_slice_pos], dy_hat_real[self.mode_truncation_slice_neg]], axis=self.data_axes[0])
                    dy_hat_imag_truncated = ops.concatenate([dy_hat_imag[self.mode_truncation_slice_pos], dy_hat_imag[self.mode_truncation_slice_neg]], axis=self.data_axes[0])

                    # compute gradients for weights, shape = (ch_in, mx, my, ch_out)
                    dw_real = ops.einsum(self.einsum_op_backprop_weights, dy_hat_real_truncated, x_hat_real_truncated) + ops.einsum(self.einsum_op_backprop_weights, dy_hat_imag_truncated, x_hat_imag_truncated)
                    dw_imag = ops.einsum(self.einsum_op_backprop_weights, dy_hat_real_truncated, x_hat_imag_truncated) - ops.einsum(self.einsum_op_backprop_weights, dy_hat_imag_truncated, x_hat_real_truncated)

                    if self.use_bias:
                        # compute gradients for bias, shape = (ch_out, )
                        db = ops.einsum(self.einsum_op_backprop_bias, dy_hat_real_truncated + dy_hat_imag_truncated)

                    # compute gradient for inputs, shape = (None, mx, my, ch_in)
                    dx_hat_real_truncated = ops.einsum(self.einsum_op_backprop_x, dy_hat_real_truncated, self.real_weights) + ops.einsum(self.einsum_op_backprop_x, dy_hat_imag_truncated, self.imag_weights)
                    dx_hat_imag_truncated = ops.einsum(self.einsum_op_backprop_x, dy_hat_imag_truncated, self.real_weights) - ops.einsum(self.einsum_op_backprop_x, dy_hat_real_truncated, self.imag_weights)

                    # pad for ifft, shape = (None, x, y, ch_in)
                    # now we have (None, my//2, mx, ch_out)
                    dx_hat_real_truncated_pos, dx_hat_real_truncated_neg = ops.split(dx_hat_real_truncated, indices_or_sections=2, axis=self.data_axes[0])
                    dx_hat_imag_truncated_pos, dx_hat_imag_truncated_neg = ops.split(dx_hat_imag_truncated, indices_or_sections=2, axis=self.data_axes[0])

                    # now we have (None, y // 2, x, ch_out)
                    dx_hat_real_pos = ops.pad(dx_hat_real_truncated_pos, pad_width=self.pad_width_pos)
                    dx_hat_real_neg = ops.pad(dx_hat_real_truncated_neg, pad_width=self.pad_width_neg)
                    dx_hat_imag_pos = ops.pad(dx_hat_imag_truncated_pos, pad_width=self.pad_width_pos)
                    dx_hat_imag_neg = ops.pad(dx_hat_imag_truncated_neg, pad_width=self.pad_width_neg)

                    dx_hat_real = ops.concatenate([dx_hat_real_pos, dx_hat_real_neg], axis=self.data_axes[0])
                    dx_hat_imag = ops.concatenate([dx_hat_imag_pos, dx_hat_imag_neg], axis=self.data_axes[0]) 

                    # apply irfft, shape = (None, x, y, ch_in)
                    dx = self.irfft((dx_hat_real, dx_hat_imag))
                    if self.use_bias:
                        return dx, [db, dw_real, dw_imag]
                    
                    return dx, [dw_real, dw_imag]

                return y, backprop
                
            return forward(x)

elif backend() == "jax":
    @saving.register_keras_serializable(package="KerasAddon.Layers.FNO", name="SpectralConv2D")
    class SpectralConv2D(BaseSpectralConv2D):
        def call(self, x: KerasTensor) -> KerasTensor:
            # forward pass, shape = (None, x, y, ch_in)
            x_hat_real, x_hat_imag = self.rfft(x)

            # reduce to relevant modes, shape = (None, mx, my, ch_in)
            x_hat_real_reduced = ops.concatenate([x_hat_real[self.mode_truncation_slice_pos], x_hat_real[self.mode_truncation_slice_neg]], axis=self.data_axes[0])
            x_hat_imag_reduced = ops.concatenate([x_hat_imag[self.mode_truncation_slice_pos], x_hat_imag[self.mode_truncation_slice_neg]], axis=self.data_axes[0])

            # apply weights to spectral convolution, shape = (None, my, mx, ch_out)
            y_hat_real_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_reduced, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_reduced, self.imag_weights)
            y_hat_imag_truncated = ops.einsum(self.einsum_op_forward, x_hat_real_reduced, self.real_weights) - ops.einsum(self.einsum_op_forward, x_hat_imag_reduced, self.imag_weights)

            # now we have (None, my//2, mx, ch_out)
            y_hat_real_truncated_pos, y_hat_real_truncated_neg = ops.split(y_hat_real_truncated, indices_or_sections=2, axis=self.data_axes[0])
            y_hat_imag_truncated_pos, y_hat_imag_truncated_neg = ops.split(y_hat_imag_truncated, indices_or_sections=2, axis=self.data_axes[0])

            # now we have (None, y // 2, x, ch_out)
            y_hat_real_pos = ops.pad(y_hat_real_truncated_pos, pad_width=self.pad_width_pos)
            y_hat_real_neg = ops.pad(y_hat_real_truncated_neg, pad_width=self.pad_width_neg)
            y_hat_imag_pos = ops.pad(y_hat_imag_truncated_pos, pad_width=self.pad_width_pos)
            y_hat_imag_neg = ops.pad(y_hat_imag_truncated_neg, pad_width=self.pad_width_neg)

            y_hat_real = ops.concatenate([y_hat_real_pos, y_hat_real_neg], axis=self.data_axes[0])
            y_hat_imag = ops.concatenate([y_hat_imag_pos, y_hat_imag_neg], axis=self.data_axes[0])    

            # add bias, shape = (None, mx, my, ch_out)
            if self.use_bias:
                y_hat_real += self.bias
                y_hat_imag += self.bias

            # reconstruct y via irfft, shape = (None, x, y, ch_out)
            y = self.irfft((y_hat_real, y_hat_imag))

            return y
  
else:
    raise RuntimeError(f"Spectral Convolution is only defined for keras backends 'tensorflow' and 'jax', received {backend()}")