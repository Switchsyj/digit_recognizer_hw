import input_data_tensorflow as idt
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import argparse
import cv2


mnist = idt.read_data_sets("MNIST_data/", one_hot=True)     # load the image


def create_placeholders(n_x, n_y):
    """
    :param n_x:
    :param n_y:
    :return: X(784 * sample) Y(10 * sample)
    """
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(tf.truncated_normal(shape=[layer_dims[l], layer_dims[l - 1]]),
                                               name=('W'+str(l)))
        parameters['b' + str(l)] = tf.Variable(tf.truncated_normal(shape=[layer_dims[l], 1],
                                                                   name=('b'+str(l))))
    return parameters


def initialize_parameters_l2reg(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(tf.truncated_normal(shape=[layer_dims[l], layer_dims[l - 1]]),
                                               name=('W'+str(l)))
        parameters['b' + str(l)] = tf.Variable(tf.truncated_normal(shape=[layer_dims[l], 1]),
                                               name=('b'+str(l)))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W' + str(l)])        # add into L2 collection
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['b' + str(l)])
    return parameters


def forward_propagation(X, parameters):
    A = {}
    Z = {}
    A['A0'] = X
    L = len(parameters) // 2
    for l in range(1, L):  # forward prop
        Z['Z' + str(l)] = tf.add(tf.matmul(parameters['W' + str(l)], A['A' + str(l - 1)]), parameters['b' + str(l)])
        A['A' + str(l)] = tf.nn.relu(Z['Z' + str(l)])
    Z['Z' + str(L)] = tf.add(tf.matmul(parameters['W' + str(L)], A['A' + str(L - 1)]), parameters['b' + str(L)])
    return Z


def compute_cost(Z, Y, layer_dims):
    L = len(layer_dims)
    logits = tf.transpose(Z['Z' + str(L-1)])
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def val_compute_cost(X, Y, parameters):
    A = {}
    Z = {}
    A['A0'] = X
    L = len(parameters) // 2
    for l in range(1, L):  # forward prop
        Z['Z' + str(l)] = tf.add(tf.matmul(parameters['W' + str(l)], A['A' + str(l - 1)]), parameters['b' + str(l)])
        A['A' + str(l)] = tf.nn.relu(Z['Z' + str(l)])
    Z['Z' + str(L)] = tf.add(tf.matmul(parameters['W' + str(L)], A['A' + str(L - 1)]), parameters['b' + str(L)])
    logits = tf.transpose(Z['Z' + str(L)])
    labels = tf.transpose(Y)
    val_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return val_cost


# def return_wrong_pre(X, Z, L, Y):
#     A = np.exp(Z['Z' + str(L - 1)]) / np.sum(np.exp(Z['Z' + str(L - 1)]))
#     equals =np.equal(np.argmax(A, axis=0), Y)
#     for i in range(len(equals)):
#         if equals[i] == False:
#             cv2.imshow()


def deep_neural_network(train_x, train_y, test_x, test_y, val_x, val_y,
                        learning_rate=0.009, num_epochs=50, minibatch_size=64,
                        layer_dims=(784, 500, 300, 10), regularization_rate=0.0001):     # train_x(pixels, sample_num), train_y(classes, sample_num)
    ops.reset_default_graph()
    (n_x, m) = train_x.shape
    n_y = train_y.shape[0]
    L = len(layer_dims)
    global_epoch = tf.Variable(0, name='global_epoch', trainable=False)

    X, Y = create_placeholders(n_x, n_y)
    # parameters = initialize_parameters(layer_dims)      # initialize parameters
    parameters = initialize_parameters_l2reg(layer_dims)        # initialize parameters with l2 reg

    Z = forward_propagation(X, parameters)        # forward prop
    cost = compute_cost(Z, Y, layer_dims)     # compute cost

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)     # add l2 reg loss (cost + reg_loss)
    reg_loss = tf.contrib.layers.apply_regularization(regularizer)
    cost = cost + reg_loss

    val_cost = val_compute_cost(X, Y, parameters)

    learning_rate_decay = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_epoch,
                                                     decay_steps=600, decay_rate=0.95,
                                                     staircase=False)      # learning rate decay
    add_global = global_epoch.assign_add(1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_decay).minimize(cost)      # Adam optimizer
    init = tf.global_variables_initializer()        # initializer

    costs = []
    val_costs = []
    max_val_acc = 0
    saver = tf.train.Saver(max_to_keep=1)  # save the best model arg max_to_keep
    with tf.Session() as sess:
        sess.run(init)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(Z['Z' + str(L - 1)], axis=0)),
                                                   tf.argmax(Y)), "float"))
        for steps in range(num_epochs):
            minibatches = idt.random_mini_batches(train_x, train_y, minibatch_size)
            epoch_cost = 0
            for minibatch in minibatches:
                minibatch_x, minibatch_y = minibatch
                decay = sess.run([add_global, learning_rate_decay])
                _, mini_cost = sess.run([optimizer, cost],
                                        feed_dict={X: minibatch_x, Y: minibatch_y})
                epoch_cost += mini_cost
            epoch_cost = epoch_cost / int(m / minibatch_size)

            val_acc = accuracy.eval({X: val_x, Y: val_y})
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                saver_path = saver.save(sess, './model/mnist.ckpt', global_step=steps+1)        # address ckpt/minist.ckpt-steps
                print(saver_path)

            val = sess.run([val_cost], feed_dict={X: val_x, Y: val_y})  # validation costs
            if steps % 5 == 0:
                print("cost after {} : {}, decay learning rate {}".format(steps, epoch_cost, decay[1]))
                costs.append(np.squeeze(epoch_cost))
                val_costs.append(np.squeeze(val))       # validation_costs append

        plt.plot(costs, "r-.")
        plt.plot(val_costs, "b-.")
        plt.xlabel("iterations / per five its")
        plt.ylabel("costs")
        plt.title("learning rate decay start= 0.009 training costs and validation cost")
        plt.show()

        parameter_return = sess.run(parameters)
        print("train accu", accuracy.eval({X: train_x, Y: train_y}))
        print("test accu", accuracy.eval({X: test_x, Y: test_y}))

    return parameter_return


def get_img_contour_thresh(img):        # return contours
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 转灰度
    blur = cv2.GaussianBlur(gray, (35, 35), 0)      # 模糊
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1,
                    help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1,
                    help="(optional) whether or not pre-trained model should be loaded")        # where to load models
    args = vars(ap.parse_args())

    parameter_print = {}
    if args["load_model"] < 0:      # training step default = -1
        parameter_print = deep_neural_network(mnist.train.images.T, mnist.train.labels.T,
                            mnist.test.images.T, mnist.test.labels.T,
                            mnist.validation.images.T, mnist.validation.labels.T)
        print(parameter_print)
    else:
        result_show = '?'
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph("./model/mnist.ckpt-36.meta")  # change iterations!!!
            new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
            # tf.summary.FileWriter('./log/', sess.graph)       # save graph structure as image
            print(sess.run('b3:0'))

            graph = tf.get_default_graph()
            # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]      # print tensor name list
            # print(tensor_name_list)
            X = graph.get_tensor_by_name('Placeholder:0')
            Z = graph.get_tensor_by_name('Add_2:0')

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            while (cap.isOpened()):
                ret, img = cap.read()
                img, contours, thresh = get_img_contour_thresh(img)
                if len(contours) > 0:
                    contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(contour) > 2500:
                        # print(predict(w_from_model,b_from_model,contour))
                        x, y, w, h = cv2.boundingRect(contour)
                        # newImage = thresh[y - 5:y + h + 5, x - 5:x + w +5]
                        newImage = thresh[y:y + h, x:x + w]
                        newImage = cv2.copyMakeBorder(newImage, 100, 40, 40, 30, cv2.BORDER_CONSTANT, value=0)       # add paddings to translate the image to the middle
                        newImage = cv2.resize(newImage, (28, 28), interpolation=cv2.INTER_LINEAR)  # 28*28 pixels
                        pre_image = np.array(newImage)

                        pre_image = pre_image.flatten()
                        pre_image = pre_image.reshape(pre_image.shape[0], 1)
                        mean = np.mean(pre_image)
                        pre_image = (pre_image - mean) / 255.0

                        y_hat = tf.nn.softmax(Z, axis=0)
                        result = tf.argmax(y_hat)
                        print(sess.run([result], feed_dict={X: pre_image}))
                        result_show = result.eval({X: pre_image})

                x, y, w, h = 0, 0, 300, 300
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "deep_neural_network prediction : " + str(result_show), (10, 320),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)     # print result on camera image

                cv2.imshow("Frame", img)
                cv2.imshow("Contours", thresh)
                cv2.imshow("processor", newImage)
                cv2.imshow("compress", cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_LINEAR))
                k = cv2.waitKey(5)  # return esc
                if k == 27:
                    break

    # print(mnist.train.images.shape, mnist.train.labels.shape,
    #         mnist.validation.images.shape, mnist.validation.labels.shape,
    #         mnist.test.images.shape, mnist.test.labels.shape)     # image shape(55000, 784), label shape(55000, 10)
    # print(mnist.train.images[1, :].reshape(28, 28))
