# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 2


def func2():
    return 74.28


def func3():
    return 'kictu'


def func4():
    return (34, 22, 55)


def func5():
    return {'glplm': 87, 'lassq': 20, 'ktjwl': 1}


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
