from utils.embeddings import Embedding, GloveEmbedding
from utils.datasets import DataSet, SquadDataSet, MacroDataSet, BabiDataSet, CombinedDataSet
import random
from config import Config


def process_embedding():
    embeddings = GloveEmbedding('data/glove.840B.300d.txt')
    embeddings.load()
    embeddings.save(Config.data_embedding)


def process_squad():
    embeddings = Embedding()
    embeddings.load(Config.data_embedding)

    data = SquadDataSet(embeddings.words)
    data.load('data/squad/train-v1.1.json')
    data.save('data/squad-train')

    data = SquadDataSet(embeddings.words)
    data.load('data/squad/dev-v1.1.json')
    data.save('data/squad-dev')


def process_macro():
    embeddings = Embedding()
    embeddings.load(Config.data_embedding)

    data = MacroDataSet(embeddings.words)
    data.load('data/macro/train_v1.1.json')
    data.save('data/macro-train')

    data = MacroDataSet(embeddings.words)
    data.load('data/macro/dev_v1.1.json')
    data.save('data/macro-dev')


def random_show():
    embeddings = Embedding()
    embeddings.load(Config.data_embedding)

    data = CombinedDataSet()
    data.load(['data/macro-train', 'data/squad-train'])

    def _extract_answer(_answer, _passage, _word_index):
        _answer_words = []
        for i in range(_answer[0], _answer[1] + 1):
            _answer_words.append(_word_index[_passage[i]])
        return " ".join(_answer_words)

    def _restore_sentence(_sentence, _word_index):
        _answer_words = []
        for i in range(len(_sentence)):
            _answer_words.append(_word_index[_sentence[i]])
        return " ".join(_answer_words)

    def debug(inx):
        ground_truth = data.data[inx][4]

        ans = _extract_answer(ground_truth, data.data[inx][0], embeddings.word_index)
        pas = _restore_sentence(data.data[inx][0], embeddings.word_index)
        que = _restore_sentence(data.data[inx][2], embeddings.word_index)
        print(pas)
        print(que)
        print(ans)
        print()

    for _ in range(20):
        debug(random.randint(0, len(data.data)))


if __name__ == '__main__':
    process_embedding()
    process_squad()
    process_macro()
    random_show()


    exit()
    process_macro()



