# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (18, 19, 89)


def func2():
    return 'qjtbg'


def func3():
    return [41, 35, 38]


def func4():
    return {'ehphm': 77, 'oerfk': 64, 'rqiqb': 22}


def func5():
    return 95


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
