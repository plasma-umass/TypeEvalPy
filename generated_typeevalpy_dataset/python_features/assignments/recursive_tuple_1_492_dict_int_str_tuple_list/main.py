# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'fgted': 27, 'wagyr': 71, 'agrrx': 6}


def func2():
    return 70


def func3():
    return 'lxsic'


def func4():
    return (8, 26, 51)


def func5():
    return [79, 74, 17]


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
