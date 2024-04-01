import os
import numpy as np
from keras.datasets import imdb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .log import logger

def decode_review(data, index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data[index]])
    return decoded_review

def plot_confusion_matrix(path2metrics, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(path2metrics, 'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    logger.info('Testing utils...')