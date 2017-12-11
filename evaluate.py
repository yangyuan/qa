from collections import Counter
import string
import re
from config import Config


def scores(index, ground_truth, passage, dict_):
    def _extract_answer(_answer, _passage, _word_index):
        _answer_words = []
        for i in range(_answer[0], _answer[1] + 1):
            _answer_words.append(_word_index[_passage[i]])
        return " ".join(_answer_words)

    answer_predict = _extract_answer(index, passage, dict_.word_index)
    answer_ground_truth = _extract_answer(ground_truth, passage, dict_.word_index)
    if Config.debug:
        print('"%s" vs "%s"' % (answer_ground_truth, answer_predict))
    f1 = f1_score(answer_predict, answer_ground_truth)
    em = exact_match_score(answer_predict, answer_ground_truth)
    return f1, em


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
