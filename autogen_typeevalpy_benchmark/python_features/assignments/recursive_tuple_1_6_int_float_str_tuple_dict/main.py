# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 99


def func2():
    return 77.13


def func3():
    return 'gvinc'


def func4():
    return (48, 92, 97)


def func5():
    return {'kqgtn': 99, 'udlxx': 2, 'ynryl': 92}


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
