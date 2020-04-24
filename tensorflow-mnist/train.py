import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Neural_Network:
    """
    MNIST学習のためのニューラルネットワーク
    """

    def __init__(self, network="cnn", sess=None, lr=0.01, epoch=10000, batch_size=100, interval=100):
        self.network = network
        self.lr = lr
        self.sess = tf.Session() if sess is None else sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.interval = interval

        self.x = tf.placeholder("float", [None, 784])
        self.y = tf.placeholder("float", [None, 10])

        # 計算グラフを定義
        self.inference = self._net()
        self.loss = self._loss(self.inference)
        self._train(self.loss)

    def _net(self):
        """
        ネットワークを定義
        hidden units = 100
        activation = relu
        """
        
        if self.network == "cnn":
            x_reshape = tf.reshape(self.x, [-1, 28, 28, 1])
            l1 = tf.layers.conv2d(x_reshape, filters=32, kernel_size=(3, 3), activation="relu")
            l1_flat = tf.layers.flatten(l1)
            output = tf.layers.dense(inputs=l1_flat, units=10, activation=None)
        elif self.network == "dense":
            l1 = tf.layers.dense(inputs=self.x, units=100, activation=tf.nn.relu)
            output = tf.layers.dense(inputs=l1, units=10, activation=None)

        return output
    
    def _loss(self, x):
        """
        損失関数を定義
        """
        
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=self.y)

        return loss
    
    def _train(self, x):
        """
        最適化手法を定義
        """
        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(x)
    
    def fit(self):
        """
        学習を実行する
        """

        init = tf.global_variables_initializer()
        self.sess.run(init)
        mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
        
        losses = []
        for i in range(self.epoch):
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})

            # intervalごとにlossとaccuracyを出力
            if i % self.interval == 0:
                loss = np.mean(self.sess.run(self.loss, feed_dict={self.x: batch_xs, self.y: batch_ys}))
                losses.append(loss)
                correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                accuracy_value = self.sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys})
                print("step: {}, loss: {:5f}, acc: {:5f}".format(str(i), loss, accuracy_value))
        
        correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy: {}".format(self.sess.run(accuracy, feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels})))
        
        plt.plot(np.arange(0, self.epoch, self.interval), losses)
        plt.savefig("results/loss.png")

if __name__ == "__main__":
    n = Neural_Network()
    n.fit()