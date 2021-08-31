import tensorflow as tf
import losses
'''
input space is R^d 
label space is {1,...,k}, y is corresponding to the label (Here k is 10)
model f : R^d -> R^k and outputs k logits
the inputs for Ldam loss is x,y 
cls_num_list is [n1,n2,...,nk] -> the number of instances of each classes
'''
#cls_num_list = [0,1,0,0,0,0,0,0,0,0]
cls_num_list = [11, 100, 5, 17, 9, 10, 6, 3, 4, 2]
ldam = losses.LDAMLoss(cls_num_list=cls_num_list)

network_outputs = tf.constant(
    [[.3, .7, 0, 0, 0, 0, 0, 0, 0, 0], [.3, .7, 0, 0, 0, 0, 0, 0, 0, 0]])
targets = tf.one_hot([1, 0], len(cls_num_list))

# network_outputs = tf.constant(
#     [[.3, .7, 0, 0, 0, 0, 0, 0, 0, 0]])
# targets = tf.constant([1])

loss = ldam(targets, network_outputs)
print("loss is", loss)


# import tensorflow.python.ops.numpy_ops.np_config
# np_config.enable_numpy_behavior()


# cls_num_list = [10, 1]
# ldam = losses.LDAMLoss(cls_num_list=cls_num_list)

# network_outputs = tf.constant([[.5, .5]])
# targets = tf.constant([[1, 0]])

# loss = ldam(network_outputs, targets)
# print(loss)
