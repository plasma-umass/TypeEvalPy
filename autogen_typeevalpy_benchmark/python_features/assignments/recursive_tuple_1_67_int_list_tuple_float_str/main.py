# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 7


def func2():
    return [91, 39, 59]


def func3():
    return (84, 21, 74)


def func4():
    return 14.62


def func5():
    return 'shdft'


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
