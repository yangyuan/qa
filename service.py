from model import *


def test():
    model = Model(is_training = False)
    print("Built model")
    dict_ = Embedding()
    dict_.load('data/embeddings')
    with model.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
            EM, F1 = 0.0, 0.0
            for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                index, ground_truth, passage = sess.run([model.output_index, model.indices, model.passage_w])
                for batch in range(Params.batch_size):
                    f1, em = f1_and_EM(index[batch], ground_truth[batch], passage[batch], dict_)
                    F1 += f1
                    EM += em
            F1 /= float(model.num_batch * Params.batch_size)
            EM /= float(model.num_batch * Params.batch_size)
            print("Exact_match: {}\nF1_score: {}".format(EM,F1))


if __name__ == '__main__':
    test()