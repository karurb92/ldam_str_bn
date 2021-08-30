import tensorflow as tf
from models import strat_bn
# import numpy as np

m = strat_bn.StratBN(axis=[-1])

input_shape = [8, 32, 32, 3]
# m.build(input_shape)
_input = tf.ones(input_shape)

# pretend that we are training so that we actually
# compute the mean and var of the batch
# otherwise we would just use the moving mean and var
# which doesn't change anything because moving mean gets
# initialized as 0

out = _input

for _ in range(10):
    out = m(out, training=True)

out = m(out, training=False)

print(out)
