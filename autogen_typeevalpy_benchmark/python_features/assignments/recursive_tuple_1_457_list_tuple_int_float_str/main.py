# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [71, 7, 46]


def func2():
    return (91, 96, 76)


def func3():
    return 87


def func4():
    return 37.25


def func5():
    return 'aftba'


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
