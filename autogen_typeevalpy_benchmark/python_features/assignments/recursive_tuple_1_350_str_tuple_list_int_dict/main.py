# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'bewug'


def func2():
    return (7, 66, 78)


def func3():
    return [70, 10, 31]


def func4():
    return 69


def func5():
    return {'oingi': 10, 'delwf': 55, 'ufqnq': 93}


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
