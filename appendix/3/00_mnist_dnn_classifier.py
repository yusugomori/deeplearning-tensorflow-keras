import os
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(123)
tf.logging.set_verbosity(tf.logging.ERROR)

'''
データの生成
'''
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

N_train = 60000

X = mnist.data.astype(np.float32)
X = X / 255.0
X = X - X.mean(axis=1).reshape(len(X), 1)
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=N_train)

'''
モデル設定
'''
n_in = 784
n_hiddens = [200, 200, 200]
n_out = 10

feature_columns = \
    [tf.contrib.layers.real_valued_column('', dimension=n_in)]

model = \
    tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=n_hiddens,
        n_classes=n_out)

'''
モデル学習
'''
model.fit(x=X_train,
          y=y_train,
          steps=300,
          batch_size=250)


accuracy = model.evaluate(x=X_test,
                          y=y_test)['accuracy']
print('accuracy:', accuracy)
