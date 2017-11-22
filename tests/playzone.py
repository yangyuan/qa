import tensorflow as tf

word_embeddings = tf.constant([[1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3], [1.4, 2.4, 3.5], [1.5, 2.5, 3.5]])
char_embeddings = tf.constant([[1.01, 2.01, 3.01], [1.02, 2.02, 3.02], [1.03, 2.03, 3.03], [1.04, 2.04, 3.05], [1.05, 2.05, 3.05]])
print(word_embeddings.shape)


words = tf.constant([[1, 0, 1, 0]])
chars = tf.constant([[1, 2, 0, 2, 1, 0, 0, 3]])

word_encoding = tf.nn.embedding_lookup(word_embeddings, words)
char_encoding = tf.nn.embedding_lookup(char_embeddings, chars)

final = tf.concat((word_encoding, char_encoding), axis=2)

sess = tf.Session()
x = sess.run(final)


print(x)