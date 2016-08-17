import tensorflow as tf

class ReadfirstLSTMCell(tf.nn.rnn_cell.RNNCell):
  """Read first LSTM recurrent network cell.

  """

  def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
    """Initialize the read first LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):
      c, h = state

      with tf.variable_scope("Gates"):
        rwf = tf.nn.rnn_cell._linear([inputs, h], 3 * self._num_units, True)
        # r = read_gate, f = forget_gate, w = write_gate
        r, w, f = tf.split(1, 3, rwf)
        r = tf.sigmoid(r)
        w = tf.sigmoid(w)
        f = tf.sigmoid(f + self._forget_bias)

      with tf.variable_scope("Candidate"):
        c_hat = tf.nn.rnn_cell._linear([inputs, h, r * c], self._num_units, True)
        c_hat = self._activation(c_hat)

      new_c = f * c + w * c_hat
      new_h = self._activation(new_c) 
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

      return new_h, new_state # RNNOut, new_state