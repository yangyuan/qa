import os
import pickle
import json
import unicodedata
import spacy


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

        def word_tokenize(text):
            nlp = spacy.blank('en')
            parsed = nlp(text)
            tokens = [i.text for i in parsed]
            return tokens

        # _words = nltk.word_tokenize(content)
        _words = word_tokenize(content)
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
            return [offset, offset + window_len - 1]
        return None

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

            range_a = self._find_answer_index(words_p, words_a, len(words_p_prefix))
            if range_a is None:
                print("Ignored one answer not found in question")
                print(question)
                print(answer['text'])
                print(paragraph['context'][answer['answer_start'] - 5:answer['answer_start'] + 5 + len(answer['text'])])
                continue
            pairs.append([words_p, chars_p, words_q, chars_q, range_a])

        return pairs
