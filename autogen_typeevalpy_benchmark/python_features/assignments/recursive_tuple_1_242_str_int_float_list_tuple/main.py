# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'mbkox'


def func2():
    return 15


def func3():
    return 88.37


def func4():
    return [47, 70, 75]


def func5():
    return (42, 6, 92)


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
