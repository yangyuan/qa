import os
import pickle
import json
import unicodedata
import spacy
import re
import nltk


debug = False


class Tokenizer:
    @staticmethod
    def word_tokenize(text):
        return nltk.word_tokenize(text)
        #nlp = spacy.blank('en')
        #parsed = nlp(text)
        tokens = [i.text for i in parsed if i.text != ' ']
        return tokens

    @staticmethod
    def char_tokenize(word):
        return unicodedata.normalize('NFD', word)

    @staticmethod
    def encode_text(content, words_id):
        _words = Tokenizer.word_tokenize(content)
        return Tokenizer.encode_words(_words, words_id)

    @staticmethod
    def encode_words(_words, words_id):
        words = []
        chars = []

        for _word in _words:
            words.append(words_id.get(_word, 0))

            _chars = []
            for _char in Tokenizer.char_tokenize(_word):
                _chars.append(ord(_char))

            chars.append(_chars)
        return words, chars


class DataSet:
    def __init__(self):
        # [[words_p, chars_p, words_q, chars_q, range_a]]
        self.data = []
        pass

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "data.pkl"), 'wb') as handle:
            pickle.dump(self.data, handle)

    def load(self, folder):
        with open(os.path.join(folder, "data.pkl"), 'rb') as handle:
            self.data = pickle.load(handle)


class SquadDataSet(DataSet):
    def __init__(self, words):
        super().__init__()
        self.words = words

    def load(self, file):
        _data = json.load(open(file))
        _size = len(_data['data'])
        print(_size)
        _count = 0
        for item in _data['data']:
            _count += 1
            if _count % 10 == 0:
                print('%d/%d' % (_count, _size))
            for paragraph in item['paragraphs']:
                self.data.extend(self.extract(paragraph))

    @staticmethod
    def _find_answer_index(context, answer, offset):
        window_len = len(answer)
        if context[offset:offset + window_len] == answer:
            return [offset, offset + window_len - 1]
        return None

    def extract(self, paragraph):

        pairs = []
        words_p, chars_p = Tokenizer.encode_text(paragraph['context'], self.words)

        for qas in paragraph['qas']:
            question = qas['question']
            words_q, chars_q = Tokenizer.encode_text(question, self.words)
            # TODO: Load all unique answers
            answer = qas['answers'][0]
            # answer['answer_start']
            words_p_prefix, _ = Tokenizer.encode_text(paragraph['context'][0: answer['answer_start']], self.words)
            words_a, _ = Tokenizer.encode_text(answer['text'], self.words)

            if len(words_p) >= 300:
                print("Ignored long p")
                continue
            if len(words_q) >= 30:
                print("Ignored long q")
                continue

            range_a = self._find_answer_index(words_p, words_a, len(words_p_prefix))
            if range_a is None:
                print("Ignored one answer not found in question")
                if debug:
                    print(question)
                    print(answer['text'])
                    print(paragraph['context'][answer['answer_start'] - 5:answer['answer_start'] + 5 + len(answer['text'])])
                continue
            pairs.append([words_p, chars_p, words_q, chars_q, range_a])

        return pairs


class MacroDataSet(DataSet):
    def __init__(self, words):
        super().__init__()
        self.words = words

    def load(self, file):
        _size = 0
        with open(file, 'r', encoding='utf8') as f:
            for _ in f:
                _size += 1
        print(_size)

        _count = 0
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                _count += 1
                if _count % 1000 == 0:
                    print('%d/%d' % (_count, _size))
                _data = json.loads(line)
                self.data.extend(self.extract(_data))

    @staticmethod
    def _find_answer_index(context, answer):
        window_len = len(answer)
        find = None
        for i in range(len(context)):
            if context[i:i + window_len] == answer:
                if find is None:
                    find = [i, i + window_len - 1]
                else:
                    return None
        return find

    @staticmethod
    def _select_passage(_data):
        for item in _data['passages']:
            if item['is_selected'] == 1:
                return item['passage_text']
        return None

    def extract(self, _data):
        passage = self._select_passage(_data)
        if passage is None:
            print('no selected passage')
            return []

        question = _data['query']

        words_p, chars_p = Tokenizer.encode_text(passage, self.words)
        words_q, chars_q = Tokenizer.encode_text(question, self.words)

        if len(words_p) >= 300:
            print("Ignored long p")
            return []
        if len(words_q) >= 30:
            print("Ignored long q")
            return []

        range_a = None
        for answer in _data['answers']:
            if answer in ['Yes', 'No']:
                print("Ignored yes no question")
                return []
            _words_a, _ = Tokenizer.encode_text(answer, self.words)
            _range_a = self._find_answer_index(words_p, _words_a)
            if _range_a is not None:
                range_a = _range_a
                break

            _words_a, _ = Tokenizer.encode_text(answer[0].lower() + answer[1:], self.words)
            _range_a = self._find_answer_index(words_p, _words_a)
            if _range_a is not None:
                range_a = _range_a
                break

            if answer[-1] != '.':
                continue
            _words_a, _ = Tokenizer.encode_text(answer[:-1], self.words)
            _range_a = self._find_answer_index(words_p, _words_a)
            if _range_a is not None:
                range_a = _range_a
                break

            _words_a, _ = Tokenizer.encode_text(answer[0].lower() + answer[1:-1], self.words)
            _range_a = self._find_answer_index(words_p, _words_a)
            if _range_a is not None:
                range_a = _range_a
                break

            if answer[0:3] != 'It ':
                continue
            _words_a, _ = Tokenizer.encode_text(answer[3:], self.words)
            _range_a = self._find_answer_index(words_p, _words_a)
            if _range_a is not None:
                range_a = _range_a
                break

        if range_a is None:
            print("Ignored one answer not found in question")
            return []

        return [[words_p, chars_p, words_q, chars_q, range_a]]


class BabiDataSet(DataSet):
    def __init__(self, words):
        super().__init__()
        self.words = words

    def load(self, file):
        with open(file, 'r') as f:
            raw_pairs = self._extract_raw(f)
        self.data = self._extract(raw_pairs)

    @staticmethod
    def _extract_raw(f):
        raw_pairs = []
        passage = []
        sequence = -1
        sequence_map = dict()
        for line in f:
            is_question = line.find('\t') > 0

            words = re.findall('([a-zA-z0-9]+|[.?]|[\t])', line)

            _sequence = int(words[0])
            if _sequence < sequence:
                passage = []
                sequence = -1
                sequence_map = dict()

            if is_question:
                # question and answer
                question_end = -1
                answer_end = -1
                for i in range(1, len(words)):
                    if words[i] == '\t':
                        if question_end == -1:
                            question_end = i
                        else:
                            answer_end = i
                            break
                if question_end == -1 or answer_end == -1:
                    raise Exception()
                question = words[1:question_end]
                answer = words[question_end + 1:answer_end]
                answer_evidences = [sequence_map[int(x)] for x in words[answer_end + 1:]]

                raw_pairs.append((passage.copy(), question, answer, answer_evidences))
            else:
                # story
                story = words[1:]
                sequence_map[int(words[0])] = len(passage)
                passage.append(story)
                sequence = _sequence
        return raw_pairs

    def _extract(self, raw_pairs):
        pairs = []
        for raw_pair in raw_pairs:
            raw_passage = raw_pair[0]
            raw_answer = raw_pair[2]
            first_evidence_line_index = raw_pair[3][0]
            first_evidence_line = raw_passage[first_evidence_line_index]

            answer_index = -1
            for i in range(len(first_evidence_line)):
                for j in range(len(raw_answer)):
                    if i + j >= len(first_evidence_line):
                        break
                    if raw_answer[j] != first_evidence_line[i+j]:
                        break
                    answer_index = i
            if answer_index == -1:
                print(' '.join(raw_answer))
                print(' '.join(first_evidence_line))
                raise Exception("answer not found in the evidence line")

            passage = []
            question = []
            for i in range(len(raw_passage)):
                if i < first_evidence_line_index:
                    answer_index += len(raw_passage[i])

                passage.extend(raw_passage[i])

            words_p, chars_p = Tokenizer.encode_words(passage, self.words)
            words_q, chars_q = Tokenizer.encode_words(question, self.words)
            range_a = [answer_index, answer_index + len(raw_answer) - 1]

            pairs.append([words_p, chars_p, words_q, chars_q, range_a])

        return pairs


class SampleDataSet(DataSet):
    def __init__(self, words):
        super().__init__()
        self.words = words
        self.original_passage = []

    def load(self, passage, question, duplicate=1):
        pairs = []
        words_p, chars_p = Tokenizer.encode_text(passage, self.words)
        words_q, chars_q = Tokenizer.encode_text(question, self.words)
        range_a = [0, 0]
        for _ in range(duplicate):
            pairs.append([words_p, chars_p, words_q, chars_q, range_a])
        self.original_passage = Tokenizer.word_tokenize(passage)

        self.data = pairs


class CombinedDataSet:
    def __init__(self):
        self.data = []
        pass

    def save(self, folder):
        raise Exception('combined dataSet cannot be saved')

    def load(self, folders):
        for folder in folders:
            with open(os.path.join(folder, "data.pkl"), 'rb') as handle:
                self.data.extend(pickle.load(handle))
                print(len(self.data))
                print(folder)
