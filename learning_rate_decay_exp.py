import tensorflow as tf
import numpy as np


def exp_decay():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.009, global_step=global_step,
                                               decay_steps=10, decay_rate=0.95)
    add_global = global_step.assign_add(1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(learning_rate))
        for i in range(100):
            rate = sess.run([add_global, learning_rate])
            print(rate)


def exp_sof():
    pre_class1 = 0
    z = np.array([-9.25, 4.24, -10.38, -19.48, 10.51, -15.92, 12.99, 15.77, 11.57, 7.43])
    sofmax = tf.nn.softmax(z.T)
    pre_class = tf.argmax(sofmax)
    with tf.Session() as sess:
        sess.run([pre_class, sofmax])
        print(sofmax.eval())
        pre_class1 = pre_class.eval()
    print(pre_class1)


if __name__ == "__main__":
    # exp_decay()
    exp_sof()
