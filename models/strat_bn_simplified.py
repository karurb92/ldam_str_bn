import tensorflow as tf
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.utils import control_flow_util
from keras import backend


# Source:
# https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers


'''
test it with:
input1 = tf.constant([[1., 10., 100.], [2., 20., 200.], [
                     3., 30., 300.], [4., 40., 400.], [5., 50., 500.]])
input2 = tf.constant([[1., 0., 0.], [1., 0., 0.], [
                     1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])
strat_classes_num = 2
'''


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
        input_shape_data = tf.TensorShape(input_shape[0])
        input_shape_strat = tf.TensorShape(input_shape[1])

        if not input_shape_data.ndims:
            raise ValueError(
                f'Input has undefined rank. Received: input_shape={input_shape_data}.')
        ndims = len(input_shape_data)

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

        axis_to_dim = {x: input_shape_data.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Received input '
                                 'with shape %s. Axis value: %s' %
                                 (tuple(input_shape), self.axis))

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = [input_shape_strat[1], list(axis_to_dim.values())[0]]
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
            ]
            param_shape = [input_shape_strat[1]] + param_shape

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

        inputs_data = inputs[0]
        inputs_strat = inputs[1]

        inputs_dtype = inputs_strat.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric stability.
            # In particular, it's very easy for variance to overflow in float16 and
            # for safety we also choose to cast bfloat16 to float32.
            inputs_strat = tf.cast(inputs_strat, tf.float32)
        inputs_dtype = inputs_data.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric stability.
            # In particular, it's very easy for variance to overflow in float16 and
            # for safety we also choose to cast bfloat16 to float32.
            inputs_data = tf.cast(inputs_data, tf.float32)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs_data.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        # print(reduction_axes)

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (len(v.shape) != ndims and reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        output = tf.zeros([0, *inputs_data.shape[1:]])
        new_means = tf.zeros([0, input_shape[self.axis[0]]])
        new_variances = tf.zeros([0, input_shape[self.axis[0]]])

        for strat_class in range(inputs_strat.shape[1]):

            inputs_subdata = tf.boolean_mask(
                inputs_data, inputs_strat[:, strat_class])

            if tf.size(inputs_subdata) == 0:
                continue

            sub_gamma = self.gamma[strat_class]
            sub_beta = self.beta[strat_class]

            # tf.print(sub_gamma)
            # tf.print(sub_beta)

            scale, offset = _broadcast(sub_gamma), _broadcast(sub_beta)

            if training == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
                print('Not training so use moving mean and var')
                mean, variance = self.moving_mean[strat_class], self.moving_variance[strat_class]
            else:
                # Some of the computations here are not necessary when training==False
                # but not a constant. However, this makes the code simpler.
                keep_dims = len(self.axis) > 1
                mean, variance = self._moments(
                    tf.cast(inputs_subdata, self._param_dtype),
                    reduction_axes,
                    keep_dims=keep_dims)

            mean = tf.cast(mean, inputs_subdata.dtype)
            variance = tf.cast(variance, inputs_subdata.dtype)
            offset = tf.cast(offset, inputs_subdata.dtype)
            scale = tf.cast(scale, inputs_subdata.dtype)

            new_means = tf.concat(
                [new_means, tf.reshape(mean, (1, -1))], axis=0)
            new_variances = tf.concat(
                [new_variances, tf.reshape(variance, (1, -1))], axis=0)

            #print(mean, variance, offset, scale)

            outputs_subdata = tf.nn.batch_normalization(inputs_subdata, _broadcast(mean),
                                                        _broadcast(
                variance), offset, scale,
                self.epsilon)
            if inputs_dtype in (tf.float16, tf.bfloat16):
                outputs_subdata = tf.cast(outputs_subdata, inputs_dtype)

            output = tf.concat([output, outputs_subdata], axis=0)
            # print(outputs_subdata)

        # this could be used for partial updates of the moving mean and variances
        # however we decided to store all moving means variances in a temporary array
        # and do one final update in the end

        input_batch_size = None

        def _do_update(var, value):
            """Compute the updates for mean and variance."""
            return self._assign_moving_average(var, value, self.momentum, input_batch_size)

        def mean_update():
            return _do_update(self.moving_mean, new_means)

        def variance_update():
            return _do_update(self.moving_variance, new_variances)

        if training == True:
            self.add_update(mean_update)
            self.add_update(variance_update)

        return output
