# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (62, 23, 14)


def func2():
    return 'cnydb'


def func3():
    return 15


def func4():
    return 65.37


def func5():
    return {'qbtvb': 2, 'vjwna': 82, 'krlrt': 90}


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
