# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'dnmfm'


def func2():
    return 95.73


def func3():
    return [81, 21, 44]


def func4():
    return (81, 40, 88)


def func5():
    return {'qppni': 41, 'jrdom': 82, 'ajgyr': 61}


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
