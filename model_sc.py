# model class here
import tensorflow as tf


# class ResnetBlock(tf.keras.layers.Layer):

#     def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
#         super(ResnetBlock, self).__init__(**kwargs)

#         self.residual_layers = []
#         for i in range(num_residuals):
#             if i == 0 and not first_block:
#                 self.residual_layers.append(
#                     Residual(num_channels, use_1x1conv=True, strides=2))
#             else:
#                 self.residual_layers.append(Residual(num_channels))

#     def call(self, X):
#         for layer in self.residual_layers.layers:
#             X = layer(X)
#         return X

# str batch norm class here
