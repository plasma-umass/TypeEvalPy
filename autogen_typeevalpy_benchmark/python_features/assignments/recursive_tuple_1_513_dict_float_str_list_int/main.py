# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'fdtbg': 26, 'obhci': 85, 'omsck': 16}


def func2():
    return 61.89


def func3():
    return 'qwlfd'


def func4():
    return [18, 81, 22]


def func5():
    return 19


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
