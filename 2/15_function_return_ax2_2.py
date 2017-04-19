def f(x, a=2):
    return a * x ** 2, 2 * a * x


y, y_prime = f(1)

print(y)
print(y_prime)
