# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 94


def func2():
    return 42.46


def func3():
    return [88, 51, 17]


def func4():
    return (96, 37, 2)


def func5():
    return 'vnjzr'


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
