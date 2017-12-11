import numpy as np
from config import Params
import math
from utils.datasets import CombinedDataSet
from config import Config


def rectify_data(data):
    q_word_ids = []
    q_char_ids = []
    q_char_len = []
    q_word_len = []
    p_word_ids = []
    p_char_ids = []
    p_char_len = []
    p_word_len = []
    indices = []
    for words_p, chars_p, words_q, chars_q, range_a in data:
        if len(words_p) >= Params.max_p_len:
            raise Exception('Passage number of words should be smaller than 300.')
        if len(words_q) >= Params.max_q_len:
            raise Exception('Question number of words should be smaller than 30.')
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
    p_max_word = Params.max_p_len  # np.max(p_word_len)
    p_max_char = Params.max_char_len  # ,max_value(p_char_len))
    q_max_word = Params.max_q_len  # ,np.max(q_word_len)
    q_max_char = Params.max_char_len  # ,max_value(q_char_len))

    # pad_data
    p_word_ids = pad_data(p_word_ids, p_max_word)
    q_word_ids = pad_data(q_word_ids, q_max_word)
    p_char_ids = pad_char_data(p_char_ids, p_max_char, p_max_word)
    q_char_ids = pad_char_data(q_char_ids, q_max_char, q_max_word)

    # to numpy
    indices = np.reshape(np.asarray(indices, np.int32), (-1, 2))
    p_word_len = np.reshape(np.asarray(p_word_len, np.int32), (-1, 1))
    q_word_len = np.reshape(np.asarray(q_word_len, np.int32), (-1, 1))
    # p_char_len = pad_data(p_char_len,p_max_word)
    # q_char_len = pad_data(q_char_len,q_max_word)
    p_char_len = pad_char_len(p_char_len, p_max_word, p_max_char)
    q_char_len = pad_char_len(q_char_len, q_max_word, q_max_char)

    for i in range(p_word_len.shape[0]):
        if p_word_len[i, 0] > p_max_word:
            p_word_len[i, 0] = p_max_word
    for i in range(q_word_len.shape[0]):
        if q_word_len[i, 0] > q_max_word:
            q_word_len[i, 0] = q_max_word

    # shapes of each data
    shapes = [(p_max_word,), (q_max_word,),
              (p_max_word, p_max_char,), (q_max_word, q_max_char,),
              (1,), (1,),
              (p_max_word,), (q_max_word,),
              (2,)]

    return ([p_word_ids, q_word_ids,
             p_char_ids, q_char_ids,
             p_word_len, q_word_len,
             p_char_len, q_char_len,
             indices], shapes)


def pad_data(data, max_word):
    padded_data = np.zeros((len(data), max_word), dtype=np.int32)
    for i,line in enumerate(data):
        for j,word in enumerate(line):
            if j >= max_word:
                print("skipped a word")
                continue
            padded_data[i, j] = word
    return padded_data


def pad_char_data(data, max_char, max_words):
    padded_data = np.zeros((len(data), max_words, max_char), dtype=np.int32)
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


def load_dev():

    data = CombinedDataSet()

    if Config.babi:
        data.load(['data/babi-en-dev', 'data/babi-hn-dev'])
    else:
        data.load(['data/squad-dev', 'data/macro-dev'])

    return rectify_data(data.data)


def load_train():

    data = CombinedDataSet()
    if Config.babi:
        data.load(['data/babi-en-train', 'data/babi-hn-train'])
    else:
        data.load(['data/squad-train', 'data/macro-train', 'data/extra-train'])

    return rectify_data(data.data)


def dev_all():
    devset, shapes = load_dev()
    indices = devset[-1]

    dev_ind = np.arange(indices.shape[0], dtype=np.int32)
    np.random.shuffle(dev_ind)
    return devset, dev_ind


def train_batches(step):
    devset, shapes = load_train()

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


def extract_by_indices(_data, _indices):
    batch = []
    for x in range(9):
        batch.append(_data[x][_indices])
    return batch