import input_data
import deep_neual_network as dnn
import cv2
from matplotlib import pyplot as plt
import numpy as np


def L_layer_model(X, Y, layer_dims, learning_rate=0.009, num_iterations=300):
    costs = []
    parameters = dnn.initialize_parameters_deep(layer_dims, initialization="he")

    for i in range(num_iterations):
        AL, caches = dnn.L_model_forward(X, parameters)
        cost = dnn.compute_cost(AL, Y)
        grads = dnn.L_model_backward(AL, Y, caches)
        if (i > 800) and (i < 1500):     # learning rate decay
            alpha = 0.8 * learning_rate
        elif i >= 1500:
            alpha = 0.5 * learning_rate
        else:
            alpha = learning_rate
        parameters = dnn.update_parameters(parameters, grads, alpha)

        if i % 100 == 0:
            print("cost after {0} : {1}".format(i, cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations/per hundred times")
    plt.ylabel("cost")
    plt.title("learning rate = " + str(learning_rate))
    plt.show()
    return parameters


def predict_model(test_x, test_y, parameters):
    m = test_x.shape[1]
    num = test_y.shape[0]
    pre, _ = dnn.L_model_forward(test_x, parameters)
    pre[pre >= 0.5] = 1
    pre[pre < 0.5] = 0
    pre = (pre == test_y).astype(int)
    pre = np.sum(pre, axis=0, keepdims=True)
    pre[pre < num] = 0
    pre[pre == num] = 1
    print(pre)
    return (1 / m) * np.sum(pre)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = input_data.load_data()
    train_x_flatten = train_x.reshape(train_x.shape[0], -1).T       # preprocessing of data
    test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
    train_y = (train_y.T).astype(int)
    test_y = test_y.T.astype(int)

    train_x_flatten = train_x_flatten / 255     # standardize
    test_x_flatten = test_x_flatten / 255

    parameters = L_layer_model(train_x_flatten, train_y, (784, 100, 10))

    train_accuracy = predict_model(train_x_flatten, train_y, parameters)
    print(train_accuracy, '\n')
    test_accuracy = predict_model(test_x_flatten, test_y, parameters)
    print(test_accuracy, '\n')

    # print(train_x_flatten.shape, train_y.shape, test_x_flatten.shape, test_y.shape)
    # cv2.imshow("train_image", train_x[0])
    # cv2.waitKey(0)
