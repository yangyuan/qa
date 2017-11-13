import numpy as np
import pickle
import os
import json
import unicodedata
import nltk


class Embeddings:
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


class GloveEmbeddings(Embeddings):
    def __init__(self, word_embedding_file, char_embedding_file):
        super().__init__()
        self.word_embedding_file = word_embedding_file
        self.char_embedding_file = char_embedding_file

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


class DataSet:
    def __init__(self):
        self.data = []
        pass

    def save(self, folder):
        with open(os.path.join(folder, "data.pkl"), 'wb') as handle:
            pickle.dump(self.data, handle)

    def load(self, folder):
        with open(os.path.join(folder, "data.pkl"), 'rb') as handle:
            self.data = pickle.load(handle)


class SquadDataSet(DataSet):
    def __init__(self, words, chars):
        super().__init__()
        self.words = words
        self.chars = chars

    def load(self, file):
        _data = json.load(open(file))
        for paragraph in _data['data'][0]['paragraphs']:
            self.data.extend(self.extract(paragraph))

    def _encode(self, content):
        words = []
        chars = []

        import spacy
        nlp = spacy.blank('en')

        def tokenize_corenlp(text):
            parsed = nlp(text)
            tokens = [i.text for i in parsed]
            return tokens

        # _words = nltk.word_tokenize(content)
        _words = tokenize_corenlp(content)
        for _word in _words:
            words.append(self.words.get(_word, 0))

            _chars = []
            for _char in unicodedata.normalize('NFD', _word):
                _chars.append(self.chars.get(_char, 0))

            chars.append(_chars)
        return words, chars

    @staticmethod
    def _find_answer_index(context, answer, offset):
        window_len = len(answer)
        if context[offset:offset + window_len] == answer:
            if window_len == 1:
                return [offset, offset]
            return [offset, offset + window_len]
        raise Exception()

    def extract(self, paragraph):

        pairs = []

        words_p, chars_p = self._encode(paragraph['context'])

        for qas in paragraph['qas']:
            question = qas['question']
            words_q, chars_q = self._encode(question)
            # TODO: Load all unique answers
            answer = qas['answers'][0]
            # answer['answer_start']
            words_p_prefix, _ = self._encode(paragraph['context'][0: answer['answer_start']])
            words_a, _ = self._encode(answer['text'])

            if len(words_p) >= 300:
                print("Ignored long p")
                continue
            try:
                range_a = self._find_answer_index(words_p, words_a, len(words_p_prefix))
                pairs.append([words_p, chars_p, words_q, chars_q, range_a])
            except:
                print("Ignored one answer not found in question")
                print(question)
                print(paragraph['context'][answer['answer_start']:])
                print(paragraph['context'][answer['answer_start']-5:answer['answer_start']+15])
                print(answer['text'])

        return pairs


def step_1():
    embeddings = GloveEmbeddings('data/glove.840B.300d.txt', 'data/glove.840B.300d.char.txt')
    embeddings.load()
    embeddings.save('data/embeddings')


def step_2():
    embeddings = Embeddings()
    embeddings.load('data/embeddings')


def step_3():
    embeddings = Embeddings()
    embeddings.load('data/embeddings')
    data = SquadDataSet(embeddings.words, embeddings.chars)
    data.load('data/train-v1.1.json')
    data.save('data/train')


def step_4():
    embeddings = Embeddings()
    embeddings.load('data/embeddings')

    data = DataSet()
    data.load('data/train')

    def debug(inx):
        print(data.data[inx][0])
        print(data.data[inx][4])
        ground_truth = data.data[inx][4]

        ans = data.data[inx][0][ground_truth[0]:ground_truth[1]]
        print(embeddings.ind2word(data.data[inx][0]))
        print(embeddings.ind2word(ans))

        print(embeddings.word_embedding.shape)

    x = 0
    for k, v in embeddings.words.items():
        print(k, v)
        x += 1
        if x > 20:
            break
    x = 0
    for k, v in embeddings.chars.items():
        print(k, v)
        x += 1
        if x > 20:
            break

    print(embeddings.word_embedding.shape)
    print(embeddings.char_embedding.shape)
    print(embeddings.char_embedding)
    debug(0)
    print(len(data.data))
    print(embeddings.word_embedding)


if __name__ == '__main__':
    step_4()

