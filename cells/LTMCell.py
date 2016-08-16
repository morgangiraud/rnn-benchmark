import tensorflow as tf

class LTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, activation=tf.nn.tanh):
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates"):
                output_size = self._num_units
                rwf = tf.nn.rnn_cell._linear([inputs, state], 3 * output_size, True, 0.0)
                rwf = tf.nn.sigmoid(rwf)

                r, w, f = tf.split(1, 3, rwf)
            
            state_hat = tf.nn.rnn_cell._linear([inputs, r * state], output_size, True, 0.0)
            state_hat = self._activation(state_hat)
                
            new_state = f * state + w * state_hat

        return new_state, new_state