# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [98, 11, 31]


def func2():
    return 'vwmao'


def func3():
    return 54


def func4():
    return (13, 31, 88)


def func5():
    return 1.99


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
