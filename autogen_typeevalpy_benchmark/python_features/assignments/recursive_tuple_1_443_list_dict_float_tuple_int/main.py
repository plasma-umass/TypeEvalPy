# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [99, 3, 59]


def func2():
    return {'vkejv': 55, 'yyplb': 35, 'arlzr': 12}


def func3():
    return 8.31


def func4():
    return (83, 51, 48)


def func5():
    return 39


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
