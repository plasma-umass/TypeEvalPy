# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 30.72


def func2():
    return 'yyyuj'


def func3():
    return [26, 43, 58]


def func4():
    return (40, 39, 14)


def func5():
    return 56


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
