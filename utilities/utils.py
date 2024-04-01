import tensorflow as tf
from keras_nlp.tokenizers import Tokenizer

from keras.datasets import imdb
from .log import logger

def decode_review(data, index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data[index]])
    return decoded_review

class WhitespaceSplitterTokenizer(Tokenizer):
    def tokenize(self, inputs):
        return tf.strings.split(inputs)

    def detokenize(self, inputs):
        return tf.strings.reduce_join(inputs, separator=" ", axis=-1)

if __name__ == '__main__':
    logger.info('Testing utils...')