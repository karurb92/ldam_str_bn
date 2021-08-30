import tensorflow as tf
import losses

# import tensorflow.python.ops.numpy_ops.np_config
# np_config.enable_numpy_behavior()


cls_num_list = [10, 1]
ldam = losses.LDAMLoss(cls_num_list=cls_num_list)

network_outputs = tf.constant([[.5, .5]])
targets = tf.constant([[1, 0]])

loss = ldam(network_outputs, targets)
print(loss)
