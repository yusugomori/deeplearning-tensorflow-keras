import os
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

'''
モデルファイル用設定
'''
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

'''
データの生成
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])


'''
保存したモデルを読み込み再実験
'''

'''
モデル再設定
'''
w = tf.Variable(tf.zeros([2, 1]), name='w')
b = tf.Variable(tf.zeros([1]), name='b')

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
学習済モデルで実験
'''
# init = tf.global_variables_initializer()  # 初期化は不要
saver = tf.train.Saver()  # モデル読み込み用
sess = tf.Session()
# sess.run(init)

# モデル読み込み
saver.restore(sess, MODEL_DIR + '/model.ckpt')
print('Model restored.')

acc = accuracy.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print('accuracy:', acc)
