# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 90


def func2():
    return [93, 29, 78]


def func3():
    return 87.4


def func4():
    return {'neqbm': 55, 'ybzyd': 93, 'eoufn': 60}


def func5():
    return 'isgeg'


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
