# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'nhinf'


def func2():
    return [70, 22, 6]


def func3():
    return {'jzxeb': 55, 'fwpfb': 64, 'vgdme': 75}


def func4():
    return (42, 17, 77)


def func5():
    return 49.0


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
