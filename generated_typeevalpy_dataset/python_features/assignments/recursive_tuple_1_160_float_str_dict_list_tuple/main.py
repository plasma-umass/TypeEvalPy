# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 77.38


def func2():
    return 'eovbp'


def func3():
    return {'ernrm': 43, 'vddkm': 11, 'uxufe': 19}


def func4():
    return [3, 3, 75]


def func5():
    return (69, 69, 8)


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
