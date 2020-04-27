import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras

class Neural_Network:
    def __init__(self, x_train, y_train, x_test, y_test, epochs=5):
        self.epochs = epochs

        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0
        self.y_train = y_train
        self.y_test = y_test

        self.model = self.build_model()

    def build_model(self):
        """ネットワーク、損失関数、最適化手法を定義.
        """
        model = keras.Sequential([
            Conv2D(32, 3, activation="relu"), 
            Flatten(), 
            Dense(128, activation="relu"), 
            Dense(10, activation="softmax"), 
        ])
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss=loss, optimizer=optimizer, metrics=["acc", "acc"])

        return model

    def fit(self):
        """学習を実施.
        """
        # patience は改善が見られるかを監視するエポック数を表すパラメーター
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.epochs, validation_split = 0.2, verbose=0,
            callbacks=[early_stop]
        )

        print(pd.DataFrame(history.history))

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)
    
    c = Neural_Network(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    c.fit()