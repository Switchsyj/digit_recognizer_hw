import numpy as np
from dnn_utils import sigmoid, relu, sigmoid_backward, relu_backward, softmax, softmax_backward


def initialize_parameters_deep(layer_dims, initialization="random"):
    parameters = {}
    L = len(layer_dims)
    if initialization == "random":
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01     # initialize W, b
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    elif initialization == "he":
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])     # initialize W, b
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (W, A, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    A_prev means previous A, b. activation is activation method
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    :param X:
    :param parameters:
    :return: final AL result and caches in each layer
    """
    L = len(parameters) // 2
    caches = []
    A = X
    for l in range(1, L):
        A_prev = A
        W, b = parameters['W' + str(l)], parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
    W, b = parameters['W' + str(L)], parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, "softmax")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))      # softmax cross entropy cost
    # AL = softmax(AL)
    # cost = (-1) * np.sum(np.multiply(np.log(AL), Y), axis=0, keepdims=True)
    # # cost = -(1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)), axis=1, keepdims=True)      # cross entropy cost
    # cost = (1 / m) * np.sum(cost, axis=1, keepdims=True)
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    """
    :param dZ:
    :param cache:
    :return:dW, dA[L-1], db
    """
    W, A_prev, b = cache        # A_prev和W求导方法不同
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    :param dA:
    :param cache:
    :param activation:
    :return:dA_prev, dW, db
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    :param AL:
    :param Y:
    :param caches:
    :return: grads
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))      # cross entropy
    # dAL = -np.divide(Y, AL + 1e-7)     # softmax cross entropy
    cur_cache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL,
                                                                        cur_cache, "softmax")
    for l in reversed(range(L - 1)):
        cur_cache = caches[l]
        grads['dA' + str(l)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = linear_activation_backward(
                                                                        grads['dA' + str(l + 1)], cur_cache, "relu")
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


# if __name__ == '__main__':
    # parameters = initialize_parameters_deep([5, 4, 3])        # initialize test case
    # print(str(parameters['W1']))

    # np.random.seed(2)     # linear_activation_forward test case
    # A_prev = np.random.rand(3, 2)
    # W = np.random.randn(1, 3)
    # b = np.random.randn(1, 1)
    # A, cache = linear_activation_forward(A_prev, W, b, "relu")
    # print(A, cache)

    # np.random.randn(6)        # L_model_forward test case compute_cost test case
    # X = np.random.randn(5, 4)
    # W1 = np.random.randn(3, 5)
    # b1 = np.random.randn(3, 1)
    # W2 = np.random.randn(1, 3)
    # b2 = np.random.randn(1, 1)
    # parameters = {
    #     'W1': W1, 'b1': b1,
    #     'W2': W2, 'b2': b2,
    # }
    # AL, caches = L_model_forward(X, parameters)
    # Y = np.random.randn(1, 4)
    # cost = compute_cost(AL, Y)
    # print(AL, len(caches), cost)

    # np.random.seed(1)       # linear_backward test case
    # dZ = np.random.randn(1, 4)
    # A = np.random.randn(5, 4)
    # W = np.random.randn(1, 5)
    # b = np.random.randn(1, 1)
    # cache = (W, A, b)
    # print(linear_backward(dZ, cache))

    # np.random.seed(1)       # linear_activation_backward test case
    # dA = np.random.randn(1, 4)
    # A = np.random.rand(5, 4)
    # W = np.random.randn(1, 5)
    # b = np.random.randn(1, 1)
    # Z = np.random.randn(1, 4)
    # linear_cache = (W, A, b)
    # activation_cache = Z
    # cache = (linear_cache, activation_cache)
    # print(linear_activation_backward(dA, cache, "sigmoid"))

    # np.random.seed(3)     # L_model_backward test case
    # Y = np.random.randn(1, 2)
    # AL = np.array([[1, 0]])
    #
    # A1 = np.random.randn(4, 2)
    # W1 = np.random.randn(3, 4)
    # b1 = np.random.randn(3, 1)
    # Z1 = np.random.randn(3, 2)
    # linear_backward_cache1 = ((W1, A1, b1), Z1)
    #
    # A2 = np.random.randn(3, 2)
    # W2 = np.random.randn(1, 3)
    # b2 = np.random.randn(1, 1)
    # Z2 = np.random.randn(1, 2)
    # linear_backward_cache2 = ((W2, A2, b2), Z2)
    # print(L_model_backward(AL, Y, caches=(linear_backward_cache1, linear_backward_cache2)))
