# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 16


def func2():
    return [94, 55, 85]


def func3():
    return 33.59


def func4():
    return (80, 93, 26)


def func5():
    return 'cdbvu'


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
