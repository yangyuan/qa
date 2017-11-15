import os
import pickle
import json
import unicodedata
import spacy
import re


class Tokenizer:
    @staticmethod
    def word_tokenize(text):
        # _words = nltk.word_tokenize(content)
        nlp = spacy.blank('en')
        parsed = nlp(text)
        tokens = [i.text for i in parsed]
        return tokens

    @staticmethod
    def char_tokenize(word):
        return unicodedata.normalize('NFD', word)

    @staticmethod
    def encode_text(content, words_id, chars_id):
        _words = Tokenizer.word_tokenize(content)
        return Tokenizer.encode_words(_words, words_id, chars_id)

    @staticmethod
    def encode_words(_words, words_id, chars_id):
        words = []
        chars = []

        for _word in _words:
            words.append(words_id.get(_word, 0))

            _chars = []
            for _char in Tokenizer.char_tokenize(_word):
                _chars.append(chars_id.get(_char, 0))

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

    def _sss(self):
        pass


class SquadDataSet(DataSet):
    def __init__(self, words, chars):
        super().__init__()
        self.words = words
        self.chars = chars

    def load(self, file):
        _data = json.load(open(file))
        for paragraph in _data['data'][0]['paragraphs']:
            self.data.extend(self.extract(paragraph))

    @staticmethod
    def _find_answer_index(context, answer, offset):
        window_len = len(answer)
        if context[offset:offset + window_len] == answer:
            return [offset, offset + window_len - 1]
        return None

    def extract(self, paragraph):

        pairs = []
        words_p, chars_p = Tokenizer.encode_text(paragraph['context'], self.words, self.chars)

        for qas in paragraph['qas']:
            question = qas['question']
            words_q, chars_q = Tokenizer.encode_text(question, self.words, self.chars)
            # TODO: Load all unique answers
            answer = qas['answers'][0]
            # answer['answer_start']
            words_p_prefix, _ = Tokenizer.encode_text(paragraph['context'][0: answer['answer_start']], self.words, self.chars)
            words_a, _ = Tokenizer.encode_text(answer['text'], self.words, self.chars)

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


class BabiDataSet(DataSet):
    def __init__(self, words, chars):
        super().__init__()
        self.words = words
        self.chars = chars

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

            words_p, chars_p = Tokenizer.encode_words(passage, self.words, self.chars)
            words_q, chars_q = Tokenizer.encode_words(question, self.words, self.chars)
            range_a = [answer_index, answer_index + len(raw_answer) - 1]

            pairs.append([words_p, chars_p, words_q, chars_q, range_a])

        return pairs
