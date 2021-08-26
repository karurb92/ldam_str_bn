import tensorflow as tf
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.utils import control_flow_util
from keras import backend


# Source:
# https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers


class StratBN(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
    ):
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

        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

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

        self.moving_mean_initializer = initializers.get(
            moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer)
        self.momentum = momentum

        self.axis = axis

    def _moments(self, inputs, reduction_axes, keep_dims):
        return tf.nn.moments(inputs, reduction_axes, keepdims=keep_dims)

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed from
            # model.
            training = False
        return training

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    # build, where you know the shapes of the input tensors and can do the rest of the initialization

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError(
                f'Input has undefined rank. Received: input_shape={input_shape}.')
        ndims = len(input_shape)

        # Convert axis to list and
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        # resolve negatives values to actually axis
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError(
                    f'Invalid axis. Expected 0 <= axis < inputs.rank (with '
                    f'inputs.rank={ndims}). Received: layer.axis={self.axis}')
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % (self.axis,))

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Received input '
                                 'with shape %s. Axis value: %s' %
                                 (tuple(input_shape), self.axis))

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
            ]

        print(param_shape)

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

        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_mean_initializer,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)

        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_variance_initializer,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)

        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        def calculate_update_delta():
            decay = tf.convert_to_tensor(
                1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(inputs_size > 0, update_delta,
                                        backend.zeros_like(update_delta))
            return update_delta

        with backend.name_scope('AssignMovingAvg') as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):  # pylint: disable=protected-access
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope)

    # call, where you do the forward computation

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric stability.
            # In particular, it's very easy for variance to overflow in float16 and
            # for safety we also choose to cast bfloat16 to float32.
            inputs = tf.cast(inputs, tf.float32)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (len(v.shape) != ndims and reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        training_value = control_flow_util.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            print('Not training so use moving mean and var')
            mean, variance = self.moving_mean, self.moving_variance
        else:
            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = len(self.axis) > 1
            mean, variance = self._moments(
                tf.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = control_flow_util.smart_cond(
                training, lambda: mean,
                lambda: tf.convert_to_tensor(moving_mean))
            variance = control_flow_util.smart_cond(
                training, lambda: variance,
                lambda: tf.convert_to_tensor(moving_variance))

            new_mean, new_variance = mean, variance
            input_batch_size = None

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   input_batch_size)

            def mean_update():
                def true_branch(): return _do_update(self.moving_mean, new_mean)
                def false_branch(): return self.moving_mean
                return control_flow_util.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch(): return _do_update(self.moving_variance, new_variance)
                def false_branch(): return self.moving_variance
                return control_flow_util.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        offset = tf.cast(offset, inputs.dtype)
        scale = tf.cast(scale, inputs.dtype)

        print(mean, variance, offset, scale)

        outputs = tf.nn.batch_normalization(inputs, _broadcast(mean),
                                            _broadcast(
                                                variance), offset, scale,
                                            self.epsilon)
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
