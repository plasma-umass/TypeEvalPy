# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 48


def func2():
    return {'wvisn': 31, 'crbnu': 31, 'retdb': 2}


def func3():
    return (34, 93, 7)


def func4():
    return 'spxwt'


def func5():
    return 98.45


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
