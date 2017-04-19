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
モデル設定
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
モデル学習
'''
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # モデル保存用
sess = tf.Session()
sess.run(init)

# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

# モデル保存
model_path = saver.save(sess, MODEL_DIR + '/model.ckpt')
print('Model saved to:', model_path)
