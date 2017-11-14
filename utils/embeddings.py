import os
import pickle
import numpy as np


class Embedding:
    def __init__(self):
        self.words = dict()
        self.chars = dict()
        self.word_embedding = np.zeros((0, 0), dtype=float)
        self.char_embedding = np.zeros((0, 0), dtype=float)
        self.word_size = 0
        self.char_size = 0
        self.dimension = 0
        self.word_index = []
        self.char_index = []

        pass

    def ind2word(self, ids):
        output = []
        for i in ids:
            output.append(str(self.word_index[i]))
        return " ".join(output)

    def save(self, folder):
        with open(os.path.join(folder, "embeddings.pkl"), 'wb') as handle:
            obj = {'dimension': self.dimension,
                   'words': self.words, 'chars': self.chars,
                   'word_index': self.word_index, 'char_index': self.char_index,
                   'word_size': self.word_size, 'char_size': self.char_size}
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

            glove_map = np.memmap(os.path.join(folder, "word_embedding.np"),
                                  dtype=float, mode='w+',
                                  shape=(self.word_size, self.dimension))
            glove_map[:] = self.word_embedding
            glove_map = np.memmap(os.path.join(folder, "char_embedding.np"),
                                  dtype=float, mode='w+',
                                  shape=(self.char_size, self.dimension))
            glove_map[:] = self.char_embedding

    def load(self, folder):
        with open(os.path.join(folder, "embeddings.pkl"), 'rb') as handle:
            obj = pickle.load(handle)
            self.dimension = obj['dimension']
            self.word_size = obj['word_size']
            self.char_size = obj['char_size']
            self.words = obj['words']
            self.chars = obj['chars']
            self.word_index = obj['word_index']
            self.char_index = obj['char_index']

        self.word_embedding = np.memmap(os.path.join(folder, "word_embedding.np"),
                                        dtype=float, mode='r',
                                        shape=(self.word_size, self.dimension))
        self.char_embedding = np.memmap(os.path.join(folder, "char_embedding.np"),
                                        dtype=float, mode='r',
                                        shape=(self.char_size, self.dimension))


class GloveEmbedding(Embedding):
    def __init__(self, word_embedding_file, char_embedding_file):
        super().__init__()
        self.word_embedding_file = word_embedding_file
        self.char_embedding_file = char_embedding_file

    # noinspection PyMethodOverriding
    def load(self):
        self.char_size = 1
        with open(self.char_embedding_file, encoding='latin-1') as f:
            for line in f:
                if self.char_size == 1:
                    items = line.strip().split(' ')
                    self.dimension = len(items) - 1
                self.char_size += 1

        self.char_embedding = np.zeros((self.char_size, self.dimension), dtype=float)
        self.chars['<unk>'] = 0
        self.char_index.append('<unk>')
        with open(self.char_embedding_file, encoding='latin-1') as f:
            count = 0
            for line in f:
                count += 1
                items = line.strip().split(' ')
                self.char_embedding[count, :] = items[1:]
                self.chars[items[0]] = count
                self.char_index.append(items[0])

        self.word_size = 1
        with open(self.word_embedding_file, encoding='latin-1') as f:
            for _ in f:
                self.word_size += 1

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
