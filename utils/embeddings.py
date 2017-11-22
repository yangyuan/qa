import os
import pickle
import numpy as np


class Embedding:
    def __init__(self):
        self.words = dict()
        self.word_embedding = np.zeros((0, 0), dtype=float)
        self.word_size = 0
        self.dimension = 0
        self.word_index = []

        pass

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "embedding.pkl"), 'wb') as handle:
            obj = {'dimension': self.dimension,
                   'words': self.words, 'word_index': self.word_index, 'word_size': self.word_size}
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

            glove_map = np.memmap(os.path.join(folder, "word_embedding.np"),
                                  dtype=float, mode='w+',
                                  shape=(self.word_size, self.dimension))
            glove_map[:] = self.word_embedding

    def load(self, folder):
        with open(os.path.join(folder, "embedding.pkl"), 'rb') as handle:
            obj = pickle.load(handle)
            self.dimension = obj['dimension']
            self.word_size = obj['word_size']
            self.words = obj['words']
            self.word_index = obj['word_index']

        self.word_embedding = np.memmap(os.path.join(folder, "word_embedding.np"),
                                        dtype=float, mode='r',
                                        shape=(self.word_size, self.dimension))


class GloveEmbedding(Embedding):
    def __init__(self, word_embedding_file):
        super().__init__()
        self.word_embedding_file = word_embedding_file

    # noinspection PyMethodOverriding
    def load(self):

        self.word_size = 1
        with open(self.word_embedding_file, encoding='latin-1') as f:
            for line in f:
                self.word_size += 1
                if self.dimension == 0:
                    items = line.strip().split(' ')
                    self.dimension = len(items) - 1

        self.word_embedding = np.zeros((self.word_size, self.dimension), dtype=float)
        self.words['<unk>'] = 0
        self.word_index.append('<unk>')
        with open(self.word_embedding_file, encoding='latin-1') as f:
            count = 0
            for line in f:
                count += 1
                if count % 10000 == 0:
                    print(count)
                items = line.strip().split(' ')
                self.word_embedding[count, :] = items[1:]
                self.words[items[0]] = count
                self.word_index.append(items[0])
