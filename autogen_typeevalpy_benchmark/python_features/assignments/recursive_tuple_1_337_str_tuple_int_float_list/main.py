# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'egdeq'


def func2():
    return (16, 86, 32)


def func3():
    return 17


def func4():
    return 71.4


def func5():
    return [30, 32, 96]


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
