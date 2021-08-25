import tensorflow as tf

# Source:
# https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers


class StratBN(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(StratBN, self).__init__()

        '''
        During training (i.e. when using `fit()` or when calling the layer/model
        with the argument `training=True`), the layer normalizes its output using
        the mean and standard deviation of the current batch of inputs. That is to
        say, for each channel being normalized, the layer returns
        `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

        - `epsilon` is small constant (configurable as part of the constructor
        arguments)
        - `gamma` is a learned scaling factor (initialized as 1)
        - `beta` is a learned offset factor (initialized as 0)
        '''

        self.epsilon = 1e-3
        self.beta = 0
        self.gamma = 1

        '''
        During inference (i.e. when using `evaluate()` or `predict()`) or when
        calling the layer/model with the argument `training=False` (which is the
        default), the layer normalizes its output using a moving average of the
        mean and standard deviation of the batches it has seen during training. That
        is to say, it returns
        `gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta`.

        `self.moving_mean` and `self.moving_var` are non-trainable variables that
        are updated each time the layer in called in training mode, as such:

        - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
        - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

        As such, the layer will only normalize its inputs during inference
        *after having been trained on data that has similar statistics as the
        inference data*.
        '''

        self.moving_mean = 0
        self.moving_variance = 1
        self.momentum = 0.99

    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
         return tf.nn.moments(inputs, reduction_axes, keepdims=keep_dims)


    # build, where you know the shapes of the input tensors and can do the rest of the initialization

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError(
                f'Input has undefined rank. Received: input_shape={input_shape}.')
        ndims = len(input_shape)


    
        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
          if axis_to_dim[x] is None:
            raise ValueError('Input has undefined `axis` dimension. Received input '
                            'with shape %s. Axis value: %s' %
                            (tuple(input_shape), self.axis))

        # Single axis batch norm (most common/default use-case)
        param_shape = (list(axis_to_dim.values())[0],)


        self.gamma = self.add_weight(
            name='gamma',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            trainable=True,
            experimental_autocast=False)

        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            trainable=True,
            experimental_autocast=False)

        

    # call, where you do the forward computation

    def call(self, inputs, training=None):
        training = self._get_training_value(training)
