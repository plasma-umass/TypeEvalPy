# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (55, 55, 33)


def func2():
    return 4


def func3():
    return [30, 92, 7]


def func4():
    return 6.95


def func5():
    return 'gymry'


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
