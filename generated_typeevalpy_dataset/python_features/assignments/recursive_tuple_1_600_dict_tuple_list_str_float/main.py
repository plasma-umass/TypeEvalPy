# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'fglfl': 32, 'fekuy': 29, 'xzsqo': 29}


def func2():
    return (2, 13, 94)


def func3():
    return [33, 68, 46]


def func4():
    return 'uerdt'


def func5():
    return 50.89


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
