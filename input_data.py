import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


def onehot(y, start, end):
    ohot = OneHotEncoder()
    a = np.linspace(start, end-1, end-start)
    b = np.reshape(a, (-1, 1)).astype(np.int32)
    ohot.fit(b)
    trans_y = ohot.transform(y).toarray()
    return trans_y


def MNISTLabel_to_onehot(train_x, train_y, test_x, test_y, shuffled=True):      # change into onehot labels
    n_classes = 10
    train_y = np.reshape(train_y, [-1, 1])
    test_y = np.reshape(test_y, [-1, 1])
    train_y = onehot(train_y.astype(np.int32), 0, n_classes)
    test_y = onehot(test_y.astype(np.int32), 0, n_classes)
    if shuffled:
        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_t = shuffle(test_x, test_y)
    return train_x, train_y, test_x, test_y


def load_data():
    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, train_y, test_x, test_y = MNISTLabel_to_onehot(train_x, train_y, test_x, test_y)
    return train_x, train_y, test_x, test_y

