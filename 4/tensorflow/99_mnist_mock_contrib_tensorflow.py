import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(1234)


class DNN(object):
    def __init__(self, n_in, n_hiddens, n_out):
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

        self._x = None
        self._y = None
        self._t = None,
        self._keep_prob = None
        self._sess = None
        self._history = {
            'accuracy': [],
            'loss': []
        }

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def inference(self, x, keep_prob):
        # 入力層 - 隠れ層、隠れ層 - 隠れ層
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                input_dim = self.n_hiddens[i-1]

            self.weights.append(self.weight_variable([input_dim, n_hidden]))
            self.biases.append(self.bias_variable([n_hidden]))

            h = tf.nn.relu(tf.matmul(
                input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)

        # 隠れ層 - 出力層
        self.weights.append(
            self.weight_variable([self.n_hiddens[-1], self.n_out]))
        self.biases.append(self.bias_variable([self.n_out]))

        y = tf.nn.softmax(tf.matmul(
            output, self.weights[-1]) + self.biases[-1])
        return y

    def loss(self, y, t):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), axis=1))
        return cross_entropy

    def training(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def fit(self, X_train, Y_train,
            nb_epoch=100, batch_size=100, p_keep=0.5,
            verbose=1):
        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)

        self._x = x
        self._t = t
        self._keep_prob = keep_prob

        y = self.inference(x, keep_prob)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(y, t)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self._y = y
        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        for epoch in range(nb_epoch):
            X_, Y_ = shuffle(X_train, Y_train)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                sess.run(train_step, feed_dict={
                    x: X_[start:end],
                    t: Y_[start:end],
                    keep_prob: p_keep
                })
            loss_ = loss.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train,
                keep_prob: 1.0
            })
            accuracy_ = accuracy.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train,
                keep_prob: 1.0
            })
            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose:
                print('epoch:', epoch,
                      ' loss:', loss_,
                      ' accuracy:', accuracy_)

        return self._history

    def evaluate(self, X_test, Y_test):
        accuracy = self.accuracy(self._y, self._t)
        return accuracy.eval(session=self._sess, feed_dict={
            self._x: X_test,
            self._t: Y_test,
            self._keep_prob: 1.0
        })


if __name__ == '__main__':
    '''
    データの生成
    '''
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(mnist.data)
    N = 10000  # MNISTの一部を使う
    indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

    X = mnist.data[indices]
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    '''
    モデル設定
    '''
    model = DNN(n_in=len(X[0]),
                n_hiddens=[200, 200, 200],
                n_out=len(Y[0]))

    '''
    モデル学習
    '''
    model.fit(X_train, Y_train,
              nb_epoch=30,
              batch_size=200,
              p_keep=0.5)

    '''
    予測精度の評価
    '''
    accuracy = model.evaluate(X_test, Y_test)
    print('accuracy: ', accuracy)
