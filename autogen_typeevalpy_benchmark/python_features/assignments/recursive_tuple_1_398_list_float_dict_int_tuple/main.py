# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [68, 5, 79]


def func2():
    return 44.99


def func3():
    return {'qjgrq': 15, 'qbwid': 47, 'rluzf': 62}


def func4():
    return 43


def func5():
    return (70, 29, 70)


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
