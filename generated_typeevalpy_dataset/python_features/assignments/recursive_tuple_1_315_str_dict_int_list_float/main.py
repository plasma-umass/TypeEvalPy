# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'hcsdg'


def func2():
    return {'bqhxt': 28, 'aatrn': 46, 'rqbee': 93}


def func3():
    return 39


def func4():
    return [54, 30, 95]


def func5():
    return 2.74


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
