# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 80.39


def func2():
    return (94, 59, 33)


def func3():
    return {'jrjzh': 97, 'uisqt': 45, 'eowqt': 100}


def func4():
    return [23, 64, 12]


def func5():
    return 43


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
