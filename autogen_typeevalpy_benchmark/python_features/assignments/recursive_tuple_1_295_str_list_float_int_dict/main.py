# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'gcgvq'


def func2():
    return [13, 59, 49]


def func3():
    return 75.16


def func4():
    return 31


def func5():
    return {'pmios': 36, 'lpcrp': 90, 'ranbe': 73}


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
