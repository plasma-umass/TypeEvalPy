# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [16, 29, 60]


def func2():
    return 'lsuns'


def func3():
    return (14, 89, 14)


def func4():
    return 64


def func5():
    return 38.39


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
