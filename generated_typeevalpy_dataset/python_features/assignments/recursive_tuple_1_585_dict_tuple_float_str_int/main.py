# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'rseiz': 24, 'kmzpz': 11, 'wapss': 67}


def func2():
    return (66, 46, 65)


def func3():
    return 79.41


def func4():
    return 'foggt'


def func5():
    return 49


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
