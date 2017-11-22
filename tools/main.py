import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from utils.embeddings import Embedding, GloveEmbedding
from utils.datasets import DataSet, SquadDataSet, MacroDataSet, BabiDataSet


def step_1():
    embeddings = GloveEmbedding('data/glove.840B.300d.txt', 'data/glove.840B.300d.char.txt')
    embeddings.load()
    embeddings.save('data/embeddings')


def step_2():
    embeddings = Embedding()
    embeddings.load('data/embeddings')


def step_3():
    embeddings = Embedding()
    embeddings.load('data/embeddings')
    data = SquadDataSet(embeddings.words, embeddings.chars)
    data.load('data/train-v1.1.json')
    data.save('data/train')


def step_4():
    embeddings = Embedding()
    embeddings.load('data/embeddings')

    data = DataSet()
    data.load('data/train')

    def _extract_answer(_answer, _passage, _word_index):
        _answer_words = []
        for i in range(_answer[0], _answer[1] + 1):
            _answer_words.append(_word_index[_passage[i]])
        return " ".join(_answer_words)

    def debug(inx):
        print(data.data[inx][0])
        print(data.data[inx][4])
        ground_truth = data.data[inx][4]

        ans = _extract_answer(ground_truth, data.data[inx][0], embeddings.word_index)
        print(ans)

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

    debug(0)
    debug(1)
    debug(2)
    debug(3)



def step_5():
    embeddings = Embedding()
    embeddings.load('data/embeddings')
    data = BabiDataSet(embeddings.words, embeddings.chars)
    data.load('data/babi/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt')
    data.save('data/babi-en-train')



def step_6():
    embeddings = Embedding()
    embeddings.load('data/embeddings')

    data = DataSet()
    data.load('data/babi-en-train')

    def _extract_answer(_answer, _passage, _word_index):
        _answer_words = []
        for i in range(_answer[0], _answer[1] + 1):
            _answer_words.append(_word_index[_passage[i]])
        return " ".join(_answer_words)

    def debug(inx):
        print(data.data[inx][0])
        print(data.data[inx][4])
        ground_truth = data.data[inx][4]

        ans = _extract_answer(ground_truth, data.data[inx][0], embeddings.word_index)
        print(ans)

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

    debug(0)
    debug(1)
    debug(2)
    debug(3)


if __name__ == '__main__':
    step_6()

