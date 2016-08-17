import tensorflow as tf

class ReadfirstMMCLSTMCell(tf.nn.rnn_cell.RNNCell):
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
    return tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape([self._num_units, self._num_units]), self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):
      c, h = state # dims: [batch_size, self._num_units, self._num_units], [batch_size, self._num_units]

      with tf.variable_scope("Gates"):
        rwf = tf.nn.rnn_cell._linear([inputs, h], 3 * self._num_units, True)
        # r = read_gate, f = forget_gate, w = write_gate
        r, w, f = tf.split(1, 3, rwf) # dims: [batch_size, self._num_units], [batch_size, self._num_units], [batch_size, self._num_units]
        w = tf.sigmoid(w)
        f = tf.sigmoid(f + self._forget_bias)
        r = tf.sigmoid(r) 

      with tf.variable_scope("Candidate"):
        c_r = tf.expand_dims(r, [2]) * c # dims: [batch_size, self._num_units, self._num_units]
        # We interpret the read gate as coefficient over the memories
        c_r = tf.reduce_mean(c_r, reduction_indices=[2]) # dims: [batch_size, self._num_units]
        c_hat = tf.nn.rnn_cell._linear([inputs, h, c_r], self._num_units, True)
        c_hat = self._activation(c_hat)

      c_f = tf.expand_dims(f, [2]) * c
      c_w = tf.expand_dims(w * c_hat, [2])
      new_c = c_f + c_w
      new_h = self._activation(tf.reduce_mean(new_c, reduction_indices=[2]) )
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

      return new_h, new_state # RNNOut, new_state