import tensorflow as tf


# =====================
# Option 1
# =====================

# https://www.tensorflow.org/guide/intro_to_modules#waiting_to_create_variables
class FlexibleDenseModule(tf.Module):
  # Note: No need for `in_features`
  def __init__(self, out_features, name=None):
    super().__init__(name=name)
    self.is_built = False
    self.out_features = out_features

  def __call__(self, x):
    # Create variables on first call.
    if not self.is_built:
      self.w = tf.Variable(
        tf.random.normal([x.shape[-1], self.out_features]), name='w')
      self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
      self.is_built = True

    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


# =====================
# Option 2
# =====================

# Source:
# https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers
class MyDenseLayer(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    # call, where you do the forward computation
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

# create
layer = MyDenseLayer(10)
_ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.