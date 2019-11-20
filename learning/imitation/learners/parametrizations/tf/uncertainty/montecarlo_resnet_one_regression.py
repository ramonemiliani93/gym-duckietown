import numpy as np
import tensorflow as tf

from .._layers import resnet_1
from ..tf_parametrization import TensorflowParametrization


class MonteCarloDropoutResnetOneRegression(TensorflowParametrization):

    def __init__(self, **kwargs):
        TensorflowParametrization.__init__(self)
        self.samples = kwargs.get('samples')
        self.keep_probability = kwargs.get('dropout')
        self.seed = kwargs.get('seed')

    def test(self, state):
        regression = TensorflowParametrization.test(self, np.repeat(state, self.samples, axis=0))
        regression = regression[0]
        return np.squeeze(np.mean(regression, axis=1)), np.squeeze(np.var(regression, axis=1))

    def architecture(self):
        model = resnet_1(self._preprocessed_state, keep_prob=self.keep_probability, seed=self.seed)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))
        model = tf.nn.dropout(model, keep_prob=self.keep_probability, seed=self.seed)

        model = tf.layers.dense(model, self.action_tensor.shape[1])

        with tf.name_scope('losses'):
            loss = tf.losses.mean_squared_error(model, self.action_tensor)
            tf.summary.scalar('mse', loss)

        return [model], loss
