# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [78, 94, 79]


def func2():
    return (7, 97, 14)


def func3():
    return 'khqvx'


def func4():
    return 17.76


def func5():
    return 52


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
