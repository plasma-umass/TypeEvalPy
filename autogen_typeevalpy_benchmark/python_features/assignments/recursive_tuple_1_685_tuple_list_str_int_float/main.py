# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (10, 33, 41)


def func2():
    return [95, 76, 73]


def func3():
    return 'dcwlg'


def func4():
    return 95


def func5():
    return 52.41


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
