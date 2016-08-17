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
                rwf = tf.nn.rnn_cell._linear([inputs, state], 3 * self._num_units, True, 0.0)
                rwf = tf.sigmoid(rwf)

                r, w, f = tf.split(1, 3, rwf)
            with tf.variable_scope("Candidate"):
                state_hat = tf.nn.rnn_cell._linear([inputs, r * state], self._num_units, True, 0.0)
                state_hat = self._activation(state_hat)
                
            new_state = f * state + w * state_hat

            # Soft bound à la batch norm
            m, v = tf.nn.moments(new_state, [0])
            new_state /= tf.sqrt(1 + v)
        return new_state, new_state # RNNOut, new_state