# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 85.69


def func2():
    return 4


def func3():
    return {'bjqma': 84, 'tspyj': 100, 'djflb': 21}


def func4():
    return 'wsnvo'


def func5():
    return (59, 94, 69)


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
