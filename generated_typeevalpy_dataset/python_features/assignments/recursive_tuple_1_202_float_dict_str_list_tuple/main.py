# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 76.64


def func2():
    return {'xipyj': 30, 'crmpy': 2, 'mfegi': 40}


def func3():
    return 'boouv'


def func4():
    return [26, 50, 23]


def func5():
    return (42, 17, 24)


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
