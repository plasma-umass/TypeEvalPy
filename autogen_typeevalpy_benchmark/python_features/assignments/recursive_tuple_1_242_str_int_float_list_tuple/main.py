# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'fsfwl'


def func2():
    return 28


def func3():
    return 15.39


def func4():
    return [38, 32, 3]


def func5():
    return (79, 62, 57)


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
