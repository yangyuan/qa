from functools import wraps
import threading
import math

from tensorflow.python.platform import tf_logging as logging

from params import Params
import numpy as np
import tensorflow as tf
from tools.main import *
from sklearn.model_selection import train_test_split

# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.
    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

def load_data(dir_):

    data = DataSet()
    data.load('data/babi-en-train')

    q_word_ids = []
    q_char_ids = []
    q_char_len = []
    q_word_len = []
    p_word_ids = []
    p_char_ids = []
    p_char_len = []
    p_word_len = []
    indices = []
    for words_p, chars_p, words_q, chars_q, range_a in data.data:
        q_word_ids.append(words_q)
        q_char_ids.append(chars_q)
        q_char_len.append([len(x) for x in chars_q])
        q_word_len.append(len(words_q))
        p_word_ids.append(words_p)
        p_char_ids.append(chars_p)
        p_char_len.append([len(x) for x in chars_p])
        p_word_len.append(len(words_p))
        indices.append(range_a)

    # Get max length to pad
    p_max_word = Params.max_p_len#np.max(p_word_len)
    p_max_char = Params.max_char_len#,max_value(p_char_len))
    q_max_word = Params.max_q_len#,np.max(q_word_len)
    q_max_char = Params.max_char_len#,max_value(q_char_len))

    # pad_data
    p_word_ids = pad_data(p_word_ids,p_max_word)
    q_word_ids = pad_data(q_word_ids,q_max_word)
    p_char_ids = pad_char_data(p_char_ids,p_max_char,p_max_word)
    q_char_ids = pad_char_data(q_char_ids,q_max_char,q_max_word)

    # to numpy
    indices = np.reshape(np.asarray(indices,np.int32),(-1,2))
    p_word_len = np.reshape(np.asarray(p_word_len,np.int32),(-1,1))
    q_word_len = np.reshape(np.asarray(q_word_len,np.int32),(-1,1))
    # p_char_len = pad_data(p_char_len,p_max_word)
    # q_char_len = pad_data(q_char_len,q_max_word)
    p_char_len = pad_char_len(p_char_len, p_max_word, p_max_char)
    q_char_len = pad_char_len(q_char_len, q_max_word, q_max_char)

    for i in range(p_word_len.shape[0]):
        if p_word_len[i,0] > p_max_word:
            p_word_len[i,0] = p_max_word
    for i in range(q_word_len.shape[0]):
        if q_word_len[i,0] > q_max_word:
            q_word_len[i,0] = q_max_word

    # shapes of each data
    shapes=[(p_max_word,),(q_max_word,),
            (p_max_word,p_max_char,),(q_max_word,q_max_char,),
            (1,),(1,),
            (p_max_word,),(q_max_word,),
            (2,)]

    return ([p_word_ids, q_word_ids,
            p_char_ids, q_char_ids,
            p_word_len, q_word_len,
            p_char_len, q_char_len,
            indices], shapes)

def get_dev():
    devset, shapes = load_data(Params.dev_dir)
    indices = devset[-1]
    # devset = [np.reshape(input_, shapes[i]) for i,input_ in enumerate(devset)]

    dev_ind = np.arange(indices.shape[0],dtype = np.int32)
    np.random.shuffle(dev_ind)
    return devset, dev_ind


def extract_by_indices(_data, _indices):
    batch = []
    for x in range(9):
        batch.append(_data[x][_indices])
    return batch


def batches(step):
    devset, shapes = load_data(Params.train_dir)

    lens = []
    size = None
    for x in range(9):
        size = devset[x].shape[0]
        lens.append(devset[x].shape[0])
    assert [size] * 9 == lens

    indices = np.asarray(range(size))
    np.random.shuffle(indices)

    k = math.ceil(size/step)

    for i in range(0, k):
        batch_indices = list(range(i*step, min((i+1)*step, size)))
        batch_indices = indices[batch_indices]
        if len(batch_indices) < step:
            break

        yield extract_by_indices(devset, batch_indices), k

def get_batch(is_training = True):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load dataset
        input_list, shapes = load_data(Params.train_dir if is_training else Params.dev_dir)
        indices = input_list[-1]

        train_ind = np.arange(indices.shape[0],dtype = np.int32)
        np.random.shuffle(train_ind)

        size = Params.data_size
        if Params.data_size > indices.shape[0] or Params.data_size == -1:
            size = indices.shape[0]
        ind_list = tf.convert_to_tensor(train_ind[:size])

        # Create Queues
        ind_list = tf.train.slice_input_producer([ind_list], shuffle=True)

        @producer_func
        def get_data(ind):
            '''From `_inputs`, which has been fetched from slice queues,
               then enqueue them again.
            '''
            return [np.reshape(input_[ind], shapes[i]) for i,input_ in enumerate(input_list)]

        data = get_data(inputs=ind_list,
                        dtypes=[np.int32]*9,
                        capacity=Params.batch_size*32,
                        num_threads=6)

        # create batch queues
        batch = tf.train.batch(data,
                                shapes=shapes,
                                num_threads=2,
                                batch_size=Params.batch_size,
                                capacity=Params.batch_size*32,
                                dynamic_pad=True)

    return batch, size // Params.batch_size

def pad_data(data, max_word):
    padded_data = np.zeros((len(data),max_word),dtype = np.int32)
    for i,line in enumerate(data):
        for j,word in enumerate(line):
            if j >= max_word:
                print("skipped a word")
                continue
            padded_data[i,j] = word
    return padded_data

def pad_char_data(data, max_char, max_words):
    padded_data = np.zeros((len(data),max_words,max_char),dtype = np.int32)
    for i,line in enumerate(data):
        for j,word in enumerate(line):
            if j >= max_words:
                  break
            for k,char in enumerate(word):
                if k >= max_char:
                    # ignore the rest of the word if it's longer than the limit
                    break
                padded_data[i,j,k] = char
    return padded_data

def pad_char_len(data, max_word, max_char):
    padded_data = np.zeros((len(data), max_word), dtype=np.int32)
    for i, line in enumerate(data):
        for j, word in enumerate(line):
            if j >= max_word:
                break
            padded_data[i, j] = word if word <= max_char else max_char
    return padded_data
