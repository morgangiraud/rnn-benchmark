import tensorflow as tf

def layer_normalization(tensor, scope=''):
    assert(len(tensor.get_shape()) == 2)

    epsilon = 1e-5
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)

    with tf.variable_scope(scope + '/layer_norm'):
        scale = tf.get_variable(
            'scale', 
            shape=[], 
            initializer=tf.constant_initializer(1.0)
        )
        shift = tf.get_variable(
            'shift',
            shape=[],
            initializer=tf.constant_initializer(0.0)
        )

    normalized_tensor = tensor - m / (v + epsilon)

    return normalized_tensor * scale + shift

