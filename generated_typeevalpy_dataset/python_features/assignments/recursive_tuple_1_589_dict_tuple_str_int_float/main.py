# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'msfbp': 81, 'popft': 95, 'syill': 82}


def func2():
    return (56, 56, 44)


def func3():
    return 'ttvab'


def func4():
    return 76


def func5():
    return 38.81


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
