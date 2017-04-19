import os
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

'''
ログファイル用設定
'''
LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')

if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)

'''
データの生成
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

'''
モデル設定
'''
w = tf.Variable(tf.zeros([2, 1]), name='w')
b = tf.Variable(tf.zeros([1]), name='b')

x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
t = tf.placeholder(tf.float32, shape=[None, 1], name='t')
y = tf.nn.sigmoid(tf.matmul(x, w) + b, name='y')

with tf.name_scope('loss'):
    cross_entropy = \
        - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
tf.summary.scalar('cross_entropy', cross_entropy)  # TensorBoard 用に登録

with tf.name_scope('train'):
    train_step = \
        tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
モデル学習
'''
init = tf.global_variables_initializer()
sess = tf.Session()

file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  # TensorBoardに対応
summaries = tf.summary.merge_all()  # 登録した変数をひとまとめにする

sess.run(init)

# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

    summary, loss = sess.run([summaries, cross_entropy], feed_dict={
        x: X,
        t: Y
    })
    file_writer.add_summary(summary, epoch)  # TensorBoard に記録
