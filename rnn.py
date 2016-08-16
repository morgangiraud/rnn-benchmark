# Basic RNN exercice from http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

# Outline of the data
# Input sequence (X): At time step t, X_t has a 50% chance of being 1 (and a 50% chance of being 0). E.g., X might be [1, 0, 0, 1, 1, 1 ... ].
# Output sequence (Y): At time step t, Y_t has a base 50% chance of being 1 (and a 50% base chance to be 0). The chance of Y_t being 1 is increased by 50% (i.e., to 100%) if X_t-3 is 1, 
# and decreased by 25% (i.e., to 25%) if X_t-8 is 1. If both X_t-3 and X_t-8 are 1, the chance of Y_t being 1 is 50% + 50% - 25% = 75%.
# Thus, there are two dependencies in the data: one at t-3 (3 steps back) and one at t-8 (8 steps back).

# This data is simple enough that we can calculate the expected cross-entropy loss for a trained RNN depending on whether or not it learns the dependencies:
# ---------------------------------------------------

import numpy as np
import tensorflow as tf
import os, time

from util import train_network, reset_graph

dir = os.path.dirname(os.path.realpath(__file__))

print("Expected cross entropy loss if the model:")
print("- learns neither dependency:", -(0.625 * np.log(0.625) +
                                      0.375 * np.log(0.375)))
print("- learns first dependency:  ",
      -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
      -0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
print("- learns both dependencies: ", -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
      - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0))

# Global config variables
def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)

    return X, np.array(Y)

def build_basic_rnn_graph_with_list(config):

    reset_graph()

    with tf.variable_scope('RNN'):
        # Placeholders
        inputs = tf.placeholder(tf.int32, shape=[config['batch_size'], None], name='Input')
        labels = tf.placeholder(tf.int32, shape=[config['batch_size'], None], name='Label')

        # Inputs
        with tf.variable_scope('RNNPreprocessing'):
            x = tf.one_hot(inputs, config['num_classes'])
            rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, config['num_steps'], x)]

        # # RNNCell
        # with tf.variable_scope('RNNCell'):
        #     W = tf.get_variable('W', shape=[config['state_size'], config['state_size']], initializer=tf.contrib.layers.xavier_initializer())
        #     U = tf.get_variable('U', shape=[config['num_classes'], config['state_size']], initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable('b', shape=[config['state_size']], initializer=tf.constant_initializer(0.0))
        #     tf.histogram_summary('W', W)
        #     tf.histogram_summary('U', U)
        #     tf.histogram_summary('b', b)

        # def RNNCell(rnn_input, state):
        #     with tf.variable_scope('RNNCell', reuse=True):
        #         W = tf.get_variable('W')
        #         U = tf.get_variable('U')
        #         b = tf.get_variable('b')        
        #         new_state = tf.tanh(tf.matmul(state, W) + tf.matmul(rnn_input, U) + b)
        #     return new_state
        # init_state = tf.zeros(shape=[config['batch_size'], config['state_size']])
        cell = tf.nn.rnn_cell.BasicRNNCell(config['state_size'])
        init_state = cell.zero_state(config['batch_size'], tf.float32)

        # # Creating the graph
        # state = init_state
        # rnn_outputs = []
        # for rnn_input in rnn_inputs:
        #     state = RNNCell(rnn_input, state)
        #     rnn_outputs.append(state)
        # final_state = rnn_outputs[-1]
        rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

        # Predictions
        with tf.variable_scope('Softmax'):
            P = tf.get_variable('P', shape=[config['state_size'], config['num_classes']], initializer=tf.contrib.layers.xavier_initializer())
            b_p = tf.get_variable('b_p', shape=[config['num_classes']], initializer=tf.constant_initializer(0.0))

            logits = [tf.matmul(rnn_output, P) + b_p for rnn_output in rnn_outputs]

        with tf.variable_scope('Loss'):
            labels_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, config['num_steps'], labels)]
            # Error
            losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label) for logit,label in zip(logits, labels_as_list)]
            total_loss = tf.reduce_mean(losses)
            tf.scalar_summary('total_loss', total_loss)

        adam = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
        train_op = adam.minimize(total_loss)

    return {
        'name': 'rnn',
        'inputs': inputs,
        'labels': labels,
        'init_state': init_state,
        'final_state': final_state,
        'total_loss': total_loss,
        'train_op': train_op
    }


config = {
    'num_steps': 10, # Number of steps before truncating backpropagation
    'batch_size': 200,
    'num_classes': 2,
    'state_size': 64,
    'learning_rate': 1e-3,
    'size': 1000000,
    'verbose': True,
    'save': True
}
data = gen_data(config['size'])

t = time.time()
graph = build_basic_rnn_graph_with_list(config)
print("It took", time.time() - t, "seconds to build the graph.")

train_network(data, graph, 10, config['num_steps'], config['batch_size'], config['verbose'], config['save'])
