import numpy as np

'''
データの生成
'''
rng = np.random.RandomState(123)

d = 2
N = 10

mean = 5

x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis=0)

'''
単純パーセプトロン
'''
w = np.zeros(d)
b = 0


def y(x):
    return step(np.dot(w, x) + b)


def step(x):
    return 1 * (x > 0)


def t(i):
    if i < N:
        return 0
    else:
        return 1


while True:
    classified = True
    for i in range(N * 2):
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w == 0) * (delta_b == 0)
    if classified:
        break

print('w:', w)
print('b:', b)

print('\nTest:')
print(y([0, 0]))  # => 0
print(y([5, 5]))  # => 1
