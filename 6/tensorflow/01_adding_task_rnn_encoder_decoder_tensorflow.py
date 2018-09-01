import numpy as np
import tensorflow as tf
from tensorflow.nn import rnn_cell
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(1234)


def inference(x, y, n_batch, is_training,
              input_digits=None, output_digits=None,
              n_hidden=None, n_out=None):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    # Encoder
    encoder = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    state = encoder.zero_state(n_batch, tf.float32)
    encoder_outputs = []
    encoder_states = []

    with tf.variable_scope('Encoder'):
        for t in range(input_digits):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            (output, state) = encoder(x[:, t, :], state)
            encoder_outputs.append(output)
            encoder_states.append(state)

    # Decoder
    decoder = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    state = encoder_states[-1]
    decoder_outputs = [encoder_outputs[-1]]

    # 出力層の重みとバイアスを事前に定義
    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    outputs = []

    with tf.variable_scope('Decoder'):
        for t in range(1, output_digits):
            if t > 1:
                tf.get_variable_scope().reuse_variables()

            if is_training is True:
                (output, state) = decoder(y[:, t-1, :], state)
            else:
                # 直前の出力を入力に用いる
                linear = tf.matmul(decoder_outputs[-1], V) + c
                out = tf.nn.softmax(linear)
                outputs.append(out)
                out = tf.one_hot(tf.argmax(out, -1), depth=output_digits)
                (output, state) = decoder(out, state)

            decoder_outputs.append(output)

    if is_training is True:
        output = tf.reshape(tf.concat(decoder_outputs, axis=1),
                            [-1, output_digits, n_hidden])

        linear = tf.einsum('ijk,kl->ijl', output, V) + c
        # linear = tf.matmul(output, V) + c
        return tf.nn.softmax(linear)
    else:
        # 最後の出力を求める
        linear = tf.matmul(decoder_outputs[-1], V) + c
        out = tf.nn.softmax(linear)
        outputs.append(out)

        output = tf.reshape(tf.concat(outputs, axis=1),
                            [-1, output_digits, n_out])
        return output


def loss(y, t):
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
                       t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                       reduction_indices=[1]))
    return cross_entropy


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.argmax(t, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == '__main__':
    def n(digits=3):
        number = ''
        for i in range(np.random.randint(1, digits + 1)):
            number += np.random.choice(list('0123456789'))
        return int(number)

    def padding(chars, maxlen):
        return chars + ' ' * (maxlen - len(chars))

    '''
    データの生成
    '''
    N = 20000
    N_train = int(N * 0.9)
    N_validation = N - N_train

    digits = 3  # 最大の桁数
    input_digits = digits * 2 + 1  # 例： 123+456
    output_digits = digits + 1  # 500+500 = 1000 以上で４桁になる

    added = set()
    questions = []
    answers = []

    while len(questions) < N:
        a, b = n(), n()  # 適当な数を２つ生成

        pair = tuple(sorted((a, b)))
        if pair in added:
            continue

        question = '{}+{}'.format(a, b)
        question = padding(question, input_digits)
        answer = str(a + b)
        answer = padding(answer, output_digits)

        added.add(pair)
        questions.append(question)
        answers.append(answer)

    chars = '0123456789+ '
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    X = np.zeros((len(questions), input_digits, len(chars)), dtype=np.integer)
    Y = np.zeros((len(questions), digits + 1, len(chars)), dtype=np.integer)

    for i in range(N):
        for t, char in enumerate(questions[i]):
            X[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers[i]):
            Y[i, t, char_indices[char]] = 1

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, train_size=N_train)

    '''
    モデル設定
    '''
    n_in = len(chars)  # 12
    n_hidden = 128
    n_out = len(chars)  # 12

    x = tf.placeholder(tf.float32, shape=[None, input_digits, n_in])
    t = tf.placeholder(tf.float32, shape=[None, output_digits, n_out])
    n_batch = tf.placeholder(tf.int32, shape=[])
    is_training = tf.placeholder(tf.bool)

    y = inference(x, t, n_batch, is_training,
                  input_digits=input_digits,
                  output_digits=output_digits,
                  n_hidden=n_hidden, n_out=n_out)
    loss = loss(y, t)
    train_step = training(loss)

    acc = accuracy(y, t)

    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
    モデル学習
    '''
    epochs = 200
    batch_size = 200

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size

    for epoch in range(epochs):
        print('=' * 10)
        print('Epoch:', epoch)
        print('=' * 10)

        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                n_batch: batch_size,
                is_training: True
            })

        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation,
            is_training: False
        })
        val_acc = acc.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation,
            is_training: False
        })

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print('validation loss:', val_loss)
        print('validation acc: ', val_acc)

        # 検証データからランダムに問題を選んで答え合わせ
        for i in range(10):
            index = np.random.randint(0, N_validation)
            question = X_validation[np.array([index])]
            answer = Y_validation[np.array([index])]
            prediction = y.eval(session=sess, feed_dict={
                x: question,
                # t: answer,
                n_batch: 1,
                is_training: False
            })
            question = question.argmax(axis=-1)
            answer = answer.argmax(axis=-1)
            prediction = np.argmax(prediction, -1)

            q = ''.join(indices_char[i] for i in question[0])
            a = ''.join(indices_char[i] for i in answer[0])
            p = ''.join(indices_char[i] for i in prediction[0])

            print('-' * 10)
            print('Q:  ', q)
            print('A:  ', p)
            print('T/F:', end=' ')
            if a == p:
                print('T')
            else:
                print('F')
        print('-' * 10)
