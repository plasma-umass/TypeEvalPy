# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 27.63


def func2():
    return [95, 4, 33]


def func3():
    return 41


def func4():
    return (13, 54, 64)


def func5():
    return 'vfbll'


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
