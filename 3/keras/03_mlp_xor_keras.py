import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(123)

'''
データの生成
'''
# XORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

'''
モデル設定
'''
model = Sequential()

# 入力層 - 隠れ層
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 隠れ層 - 出力層
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

'''
モデル学習
'''
model.fit(X, Y, epochs=4000, batch_size=4)

'''
学習結果の確認
'''
classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
