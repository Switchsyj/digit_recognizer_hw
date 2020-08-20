import numpy as np


def sigmoid(Z):
    """
    :param Z:
    :return: sigmoid cache
    """
    cache = Z
    A = 1 / (1 + np.exp(-Z))
    return A, cache


def relu(Z):
    """
    :param Z:
    :return: relu cache
    """
    cache = Z
    A = np.maximum(0, Z)
    return A, cache


def softmax(Z):
    """
    :param Z:
    :return:A, cache
    """
    cache = Z
    Z = Z - np.max(Z)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    t = 1 / (1 + np.exp(-Z))
    dZ = dA * t * (1 - t)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, cache):
    """
    :param dA: shape:activation_neurons * sample_number
    :param cache:
    :return:
    """
    Z = cache
    Z = Z - np.max(Z)
    S = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    dZ = dA * S * (1 - S)
    return dZ


# if __name__ == "__main__":
#     np.random.seed(2)
#     dA = np.random.randn(10, 5)
#     Z = np.random.randn(10, 5)
#     cache = Z
#     print(softmax_backward(dA, cache))
