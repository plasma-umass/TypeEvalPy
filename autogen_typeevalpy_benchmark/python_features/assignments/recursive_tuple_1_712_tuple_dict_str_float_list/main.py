# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (63, 38, 21)


def func2():
    return {'pvacm': 51, 'yedrp': 29, 'xjfwv': 17}


def func3():
    return 'vtzdf'


def func4():
    return 16.53


def func5():
    return [6, 39, 69]


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
