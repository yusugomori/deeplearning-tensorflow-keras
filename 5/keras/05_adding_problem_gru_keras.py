import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)


def mask(T=200):
    mask = np.zeros(T)
    indices = np.random.permutation(np.arange(T))[:2]
    mask[indices] = 1
    return mask


def toy_problem(N=10, T=200):
    signals = np.random.uniform(low=0.0, high=1.0, size=(N, T))
    masks = np.zeros((N, T))
    for i in range(N):
        masks[i] = mask(T)

    data = np.zeros((N, T, 2))
    data[:, :, 0] = signals[:]
    data[:, :, 1] = masks[:]
    target = (signals * masks).sum(axis=1).reshape(N, 1)

    return (data, target)


'''
データの生成
'''
N = 10000
T = 200
maxlen = T

X, Y = toy_problem(N=N, T=T)

N_train = int(N * 0.9)
N_validation = N - N_train


'''
モデル設定
'''
n_in = len(X[0][0])  # 2
n_hidden = 100
n_out = len(Y[0])  # 1


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)


early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)

model = Sequential()
model.add(GRU(n_hidden,
              kernel_initializer=weight_variable,
              input_shape=(maxlen, n_in)))
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error',
              optimizer=optimizer)

'''
モデル学習
'''
epochs = 1000
batch_size = 100

hist = model.fit(X, Y,
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks=[early_stopping])

'''
学習の進み具合を可視化
'''
loss = hist.history['loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(len(loss)), loss, label='loss', color='black')
plt.xlabel('epochs')
plt.show()
plt.savefig(__file__ + '.eps')
