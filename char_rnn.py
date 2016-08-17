import numpy as np
import time, os, argparse
import tensorflow as tf

from util import load_tinyshakespeare, gen_epochs, reset_graph, train_network
from cells.LTMCell import LTMCell
from cells.PseudoLSTMCell import PseudoLSTMCell
from cells.ReadfirstLSTMCell import ReadfirstLSTMCell

dir = os.path.dirname(os.path.realpath(__file__))

def build_rnn(config):
    name = config['name']
    batch_size = config.get('batch_size', 32)
    num_classes = config.get('num_classes', 2)
    num_steps = config.get('num_steps', 200)
    state_size = config.get('state_size', 100)
    lr = config.get('lr', 1e-4)
    num_layers = config.get('num_layers', 1)

    reset_graph()

    with tf.variable_scope('RNN'):
        inputs = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='Inputs_placeholder')
        labels = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='Labels_placeholder')

        with tf.variable_scope('Embedding'):
            # We are encoding each letter into state_size space
            embedding = tf.get_variable('embedding', shape=[num_classes, state_size])
            rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)

        state_is_tuple=False
        if name == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
        elif name == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(state_size)
        elif name == 'ltm':
            cell = LTMCell(state_size, activation=tf.nn.tanh)
        elif name == 'pseudolstm':
            cell = PseudoLSTMCell(state_size, activation=tf.nn.tanh)
        elif name == 'lstm':
            state_is_tuple=True
            cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=state_is_tuple)
        elif name == 'peepholelstm':
            state_is_tuple=True
            cell = tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True, state_is_tuple=state_is_tuple)
        elif name == 'readfirstlstm':
            state_is_tuple=True
            cell = ReadfirstLSTMCell(state_size)
        # elif name == 'readfirstmmclstm':
        #     state_is_tuple=True
        #     cell = ReadfirstMMCLSTMCell(state_size)
        else:
            raise ValueError("Cell %s not handled" % (name))

        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=state_is_tuple)

        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

        with tf.variable_scope('Softmax'):
            W_s = tf.get_variable('W_s', shape=[state_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_s = tf.get_variable('b_z', shape=[num_classes], initializer=tf.constant_initializer(0.0))

            rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
            labels_reshaped = tf.reshape(labels, [-1])
            logits = tf.matmul(rnn_outputs, W_s) + b_s

        with tf.variable_scope('Loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_reshaped)
            
            total_loss = tf.reduce_mean(losses)
            tf.scalar_summary('total_loss', total_loss)

        adam = tf.train.AdamOptimizer(lr)
        train_op = adam.minimize(total_loss)

    return {
        'name': name,
        'inputs': inputs,
        'labels': labels,
        'init_state': init_state,
        'final_state': final_state,
        'total_loss': total_loss,
        'train_op': train_op
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rnn", help="model name (default: %(default)s)")
    parser.add_argument("--num_steps", default="200", type=int, help="backprop truncation limit (default: %(default)s)")
    parser.add_argument("--batch_size", default="32", type=int, help="batch size (default: %(default)s)")
    parser.add_argument("--state_size", default="100", type=int, help="hidden state size (default: %(default)s)")
    parser.add_argument("--num_layers", default="1", type=int, help="How deep is the RNN (default: %(default)s)")
    parser.add_argument("--lr", default="1e-3", type=float, help="learning rate (default: %(default)s)")
    parser.add_argument("--num_epoch", default="5", type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument("--verbose", default="1", type=int, help="(default: %(default)s)")
    parser.add_argument("--save", default="1", type=int, help="(default: %(default)s)")
    
    args = parser.parse_args()

    ts = load_tinyshakespeare()
    print("Data length:", len(ts['data']))

    config = {
        'name': args.name,
        'num_classes': ts['vocab_size'],
        'num_steps': args.num_steps,
        'batch_size': args.batch_size,
        'state_size': args.state_size,
        'num_layers': args.num_layers,
        'lr': args.lr,
        'verbose': False if args.verbose == 0 else True,
        'save': False if args.save == 0 else True
    }

    t = time.time()
    graph = build_rnn(config)
    print("It took", time.time() - t, "seconds to build the graph.")

    t2 = time.time()
    train_network(ts['data'], graph, args.num_epoch, config['num_steps'], config['batch_size'], config['verbose'], config['save'])
    print("It took", time.time() - t2, "seconds to train the graph.")
