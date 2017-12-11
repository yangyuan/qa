import tensorflow as tf
from config import Params


def gated_attention(memory, inputs, states, units, batch_size, params, self_matching = False, memory_len = None, scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states,(batch_size, Params.attn_size))
        if not self_matching:
            inputs_.append(states)
        shapes = memory.get_shape().as_list()
        scores = attention(inputs_, units, weights, memory_len = memory_len, _shape=shapes)
        scores = tf.expand_dims(scores,-1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs


def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None,
                  _shape = None, _idk=False):
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs, weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.



            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and not _idk:
                outputs = tf.reshape(outputs, (shapes[0], 1, -1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0], -1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if Params.bias:
            b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now



def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen=score.shape[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)