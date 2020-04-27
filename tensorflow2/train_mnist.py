import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class Neural_Network:
    def __init__(self, x_train, y_train, x_test, y_test, epochs=5, shuffle_size=10000, batch_size=32):
        self.epochs = epochs

        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0
        self.y_train = y_train
        self.y_test = y_test

        # データセットのシャッフルおよびバッチ化
        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(shuffle_size).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        # モデルのインスタンスを作成
        self.model = CNNModel()

        # 損失関数の定義
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        # オプティマイザーの定義
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    
    # @tf.functionはPythonの一般の関数と変わらずに呼び出せる.
    # 一方、実行時にはその関数はグラフへとコンパイルされる.
    @tf.function
    def train_step(self, images, labels):
        """
        モデルの訓練.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        """
        モデルのテスト.
        """
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    
    def fit(self):
        for epoch in range(self.epochs):
            for images, labels in self.train_ds:
                self.train_step(images, labels)

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print (template.format(epoch+1,
                                self.train_loss.result(),
                                self.train_accuracy.result()*100,
                                self.test_loss.result(),
                                self.test_accuracy.result()*100))
        
            # 次のエポック用にメトリクスをリセット
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)
    
    c = Neural_Network(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    c.fit()