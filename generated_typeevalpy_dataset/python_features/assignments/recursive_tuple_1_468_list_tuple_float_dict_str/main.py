# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [99, 37, 28]


def func2():
    return (49, 9, 97)


def func3():
    return 10.82


def func4():
    return {'bhhzg': 17, 'fdbha': 58, 'eeoqn': 99}


def func5():
    return 'xurfb'


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
