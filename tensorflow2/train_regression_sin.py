"""
TensorFlow2.0を用いてsin波の予測を行う.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Regression_network:
    def __init__(self, 
                 train_dataset, 
                 train_labels, 
                 test_dataset, 
                 test_labels, 
                 epochs=1000):

        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.epochs = epochs

        self.model = self.build_model()

    def build_model(self):
        """モデルの定義.
        """

        model = keras.Sequential([
            layers.LSTM(64), 
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
        return model

    def fit(self):
        """学習を実施する関数.
        """

        # patience は改善が見られるかを監視するエポック数を表すパラメーター
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = self.model.fit(
            self.train_dataset, self.train_labels,
            epochs=self.epochs, validation_split = 0.2, verbose=0,
            callbacks=[early_stop]
        )

        print(pd.DataFrame(history.history))

    def evaluate(self):
        """モデルの評価を実施する関数.
        """

        loss, mae, mse = self.model.evaluate(self.test_dataset, self.test_labels, verbose=2)

        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

        test_predictions = self.model.predict(self.test_dataset).flatten()

        plt.plot(test_predictions)
        plt.savefig("./results/sinx.png")
        plt.cla()

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

def add_noise(T=1000, ampl=0.05):
    """sin波にノイズを付与する.
    """
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

def make_dataset(low_data, n_prev=100):
    """データを[batch, timestep, value]の形に変形.
    """

    data = []
    target = []
    maxlen = 25

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target

if __name__ == "__main__":
    n = add_noise()
    train_dataset, train_labels = make_dataset(n)

    test_n = add_noise(T=100)
    test_dataset, test_labels = make_dataset(test_n)

    c = Regression_network(
        train_dataset, 
        train_labels, 
        test_dataset, 
        test_labels, 
    )

    c.fit()
    c.evaluate()