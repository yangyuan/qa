from utils.embeddings import Embedding, GloveEmbedding
from utils.datasets import DataSet, SquadDataSet


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

