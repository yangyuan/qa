from model import Model
from utils.embeddings import Embedding
from config import Config

import tensorflow as tf

from flask import Flask, jsonify, request, redirect

import datetime
from utils.datasets import SampleDataSet
from datautils import rectify_data

import requests
from urllib.parse import quote


class DemoApp:
    def __init__(self):
        print('loading model')
        self.model = Model(Config.service_batch_size, is_training=False)
        print('loading embedding')
        dict_ = Embedding()
        dict_.load(Config.data_embedding)
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            self.session = tf.Session()
            self.dict = dict_
            print('restoring checkpoint')
            saver.restore(self.session, tf.train.latest_checkpoint(Config.service_dir))
            print("ready")

    def predict(self, samples):
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
        return answer_predict[0]


embeddings = Embedding()
embeddings.load(Config.data_embedding)


app = Flask(__name__, static_url_path='', static_folder='web')
wtf = DemoApp()


@app.route("/")
def index():
    return redirect('/index.html')


@app.route('/qa', methods=['GET', 'POST'])
def qa():
    form = request.form

    data = SampleDataSet(embeddings.words)
    data.load(form['passage'], form['question'], Config.service_batch_size)

    try:
        _x, _ = rectify_data(data.data)
        _answer = wtf.predict(_x)

        _answer_words = []
        for i in range(_answer[0], _answer[1] + 1):
            _answer_words.append(data.original_passage[i])
        answer = " ".join(_answer_words)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e.args[0])})


# another demo for what is questions.
# can work in some concepts which have Wikipedia names
@app.route('/what', methods=['GET', 'POST'])
def what():
    form = request.form

    data = SampleDataSet(embeddings.words)

    try:
        title = form['question']
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&titles="
            + quote(title))
        page = list(resp.json()['query']['pages'].values())[0]
        text = page['extract']
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to fetch the wiki page.'})

    question = 'What is ' + title + '?'

    import nltk
    text = list(nltk.word_tokenize(text))[0:256]
    data.load(' '.join(text), question, Config.service_batch_size)

    try:
        _x, _ = rectify_data(data.data)
        _answer = wtf.predict(_x)

        _answer_words = []
        for i in range(_answer[0], _answer[1] + 1):
            _answer_words.append(data.original_passage[i])
        answer = " ".join(_answer_words)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e.args[0])})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
