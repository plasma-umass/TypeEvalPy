# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'cvpom'


def func2():
    return (63, 67, 99)


def func3():
    return [10, 20, 23]


def func4():
    return {'vhqbj': 67, 'blapa': 36, 'ggiij': 68}


def func5():
    return 20


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
