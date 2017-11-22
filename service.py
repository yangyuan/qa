from model import Model
from evaluate import f1_and_EM
from utils.embeddings import Embedding
from config import Config

import tensorflow as tf

from flask import Flask, jsonify, request

import datetime
from utils.datasets import SampleDataSet
from datautils import rectify_data


class WtfApp:
    def __init__(self):
        print('loading model')
        self.model = Model(Config.service_batch_size, is_training = False)
        print('loading embedding')
        dict_ = Embedding()
        dict_.load(Config.data_embedding)
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            self.session = tf.Session()
            self.dict = dict_
            print('restoring checkpoint')
            saver.restore(self.session, tf.train.latest_checkpoint(Config.tf_batch_dir))
            print("ready")

    def xxx(self, samples):
        model = self.model
        dict_ = self.dict
        feed_dict = {model.words_p_placeholder: samples[0],
                     model.words_q_placeholder: samples[1],
                     model.chars_p_placeholder: samples[2],
                     model.chars_q_placeholder: samples[3],
                     model.len_words_p_placeholder: samples[4],
                     model.len_words_q_placeholder: samples[5],
                     model.len_chars_p_placeholder: samples[6],
                     model.len_chars_q_placeholder: samples[7],
                     model.answer_placeholder: samples[8],
                     model.word_embeddings_placeholder: dict_.word_embedding}
        print(datetime.datetime.now())
        answer_predict = self.session.run(model.output_index, feed_dict=feed_dict)
        print(datetime.datetime.now())
        F1, EM = 0.0, 0.0
        for _index in range(Config.service_batch_size):
            f1, em = f1_and_EM(answer_predict[_index], samples[8][_index], samples[0][_index], dict_)
            F1 += f1
            EM += em
        F1 /= float(Config.service_batch_size)
        EM /= float(Config.service_batch_size)
        print("\nDev_Exact_match: {}\nDev_F1_score: {}".format(EM, F1))
        return answer_predict[0]


embeddings = Embedding()
embeddings.load(Config.data_embedding)


app = Flask(__name__, static_url_path='', static_folder='web')
wtf = WtfApp()


@app.route('/qa', methods=['GET', 'POST'])
def index():
    jsxx = request.form
    print(jsxx)

    data = SampleDataSet(embeddings.words)
    data.load(jsxx['passage'], jsxx['question'], Config.service_batch_size)
    print(data.original_passage)

    xxx, _ = rectify_data(data.data)
    print(xxx)
    _answer = wtf.xxx(xxx)

    print(_answer)

    _answer_words = []
    for i in range(_answer[0], _answer[1] + 1):
        _answer_words.append(data.original_passage[i])
    answer = " ".join(_answer_words)

    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=False)
