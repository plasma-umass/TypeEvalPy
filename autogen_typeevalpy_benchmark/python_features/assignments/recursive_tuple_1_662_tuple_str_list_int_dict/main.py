# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (8, 38, 50)


def func2():
    return 'tlqii'


def func3():
    return [64, 61, 71]


def func4():
    return 51


def func5():
    return {'chktt': 29, 'zolim': 28, 'sfvgb': 99}


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
