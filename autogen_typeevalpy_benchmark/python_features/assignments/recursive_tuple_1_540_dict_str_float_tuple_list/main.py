# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'gqzsf': 33, 'ziflh': 67, 'emnre': 97}


def func2():
    return 'yzjqr'


def func3():
    return 41.07


def func4():
    return (89, 81, 82)


def func5():
    return [27, 67, 44]


def func6():
    pass


a, (b, c) = func1, (func2, func3)
i = a()
j = b()
k = c()

a, (b, (c, d)) = func1, (func2, (func3, func4))

h = d()

f, b = c, e = func5, func6

l = e()
m = f()
