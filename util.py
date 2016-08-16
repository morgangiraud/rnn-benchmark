from __future__ import division

import sys, time, os, urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

dir = os.path.dirname(os.path.realpath(__file__))

def load_tinyshakespeare():
    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file = dir + '/data/tinyshakespeare.txt'
    if not os.path.exists(file):
        urllib.request.urlretrieve(file_url, file)

    with open(file,'r') as f:
        raw_data = f.read()

    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    data = [vocab_to_idx[char] for char in raw_data]
    del raw_data

    return {
        'vocab': vocab,
        'vocab_size': vocab_size,
        'idx_to_vocab': idx_to_vocab,
        'vocab_to_idx': vocab_to_idx,
        'data': data,
    }

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length*i:batch_partition_length*(i+1)]
        data_y[i] = raw_y[batch_partition_length*i:batch_partition_length*(i+1)]

    epoch_size = batch_partition_length // num_steps
    for i in range(epoch_size):
        x = data_x[:, num_steps * i: num_steps * (i+1)]
        y = data_y[:, num_steps * i: num_steps * (i+1)]
        yield (x, y)

def gen_epochs(data, num_epochs, batch_size, num_steps):
    for i in range(num_epochs):
        if type(data) is tuple:
            yield gen_batch(data, batch_size, num_steps)
        else:
            yield reader.ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(data, graph, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    result_folder = dir + '/results/' + graph['name'] + '/' + str(int(time.time()))
    if type(data) is tuple:
        data_length = len(data[0])
    else:
        data_length = len(data)

    tf.set_random_seed(1234)    
    init_op = tf.initialize_all_variables()
    summaries = tf.merge_all_summaries()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        sw = tf.train.SummaryWriter(result_folder, sess.graph)
        global_step = 1

        for idx, epoch in enumerate(gen_epochs(data, num_epochs, batch_size, num_steps)):
            if verbose:
                print("EPOCH %d/%d" % (idx + 1, num_epochs))

            state = None
            for batch_idx, (X,Y) in enumerate(epoch):
                feed_dict = {
                    graph['inputs']: X,
                    graph['labels']: Y
                }
                if state is not None:
                    feed_dict[graph['init_state']] = state

                training_loss, state, summary, _ = sess.run([
                        graph['total_loss'],
                        graph['final_state'],
                        summaries,
                        graph['train_op']
                    ],
                    feed_dict=feed_dict
                )
                if verbose and (batch_idx + 1) % 100 == 0:
                    print("  BATCH %d/%d, loss: %f" % (batch_idx + 1, data_length // batch_size // num_steps, training_loss))

                sw.add_summary(summary, global_step)
                global_step += 1

        if save == True:
            saver.save(sess, result_folder + '/' + graph['name'] + '_weights')


# From https://raw.githubusercontent.com/vinhkhuc/MemN2N-babi-python/master/util.py
def parse_babi_task(data_files, dictionary, include_question):
    """ Parse bAbI data.

    Args:
       data_files (list): a list of data file's paths.
       dictionary (dict): word's dictionary
       include_question (bool): whether count question toward input sentence.

    Returns:
        A tuple of (story, questions, qstory):
            story (3-D array)
                [position of word in sentence, sentence index, story index] = index of word in dictionary
            questions (2-D array)
                [0-9, question index], in which the first component is encoded as follows:
                    0 - story index
                    1 - index of the last sentence before the question
                    2 - index of the answer word in dictionary
                    3 to 13 - indices of supporting sentence
                    14 - line index
            qstory (2-D array) question's indices within a story
                [index of word in question, question index] = index of word in dictionary
    """
    # Try to reserve spaces beforehand (large matrices for both 1k and 10k data sets)
    # maximum number of words in sentence = 20
    story     = np.zeros((20, 500, len(data_files) * 3500), np.int16)
    questions = np.zeros((14, len(data_files) * 10000), np.int16)
    qstory    = np.zeros((20, len(data_files) * 10000), np.int16)

    # NOTE: question's indices are not reset when going through a new story
    story_idx, question_idx, sentence_idx, max_words, max_sentences = -1, -1, -1, 0, 0

    # Mapping line number (within a story) to sentence's index (to support the flag include_question)
    mapping = None

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip().lower()
                words = line.split()

                # Story begins
                if words[0] == '1':
                    story_idx += 1
                    sentence_idx = -1
                    mapping = []

                # FIXME: This condition makes the code more fragile!
                if '?' not in line:
                    is_question = False
                    sentence_idx += 1
                else:
                    is_question = True
                    question_idx += 1
                    questions[0, question_idx] = story_idx
                    questions[1, question_idx] = sentence_idx
                    if include_question:
                        sentence_idx += 1

                mapping.append(sentence_idx)

                # Skip substory index
                for k in range(1, len(words)):
                    w = words[k]

                    if w.endswith('.') or w.endswith('?'):
                        w = w[:-1]

                    if w not in dictionary:
                        dictionary[w] = len(dictionary)

                    if max_words < k:
                        max_words = k

                    if not is_question:
                        story[k - 1, sentence_idx, story_idx] = dictionary[w]
                    else:
                        qstory[k - 1, question_idx] = dictionary[w]
                        if include_question:
                            story[k - 1, sentence_idx, story_idx] = dictionary[w]

                        # NOTE: Punctuation is already removed from w
                        if words[k].endswith('?'):
                            answer = words[k + 1]
                            if answer not in dictionary:
                                dictionary[answer] = len(dictionary)

                            questions[2, question_idx] = dictionary[answer]

                            # Indices of supporting sentences
                            for h in range(k + 2, len(words)):
                                questions[1 + h - k, question_idx] = mapping[int(words[h]) - 1]

                            questions[-1, question_idx] = line_idx
                            break

                if max_sentences < sentence_idx + 1:
                    max_sentences = sentence_idx + 1

    story     = story[:max_words, :max_sentences, :(story_idx + 1)]
    questions = questions[:, :(question_idx + 1)]
    qstory    = qstory[:max_words, :(question_idx + 1)]

    return story, questions, qstory
