# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 63


def func2():
    return 76.48


def func3():
    return (69, 11, 84)


def func4():
    return [84, 10, 69]


def func5():
    return {'avrrj': 29, 'vvjet': 96, 'samyl': 3}


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
