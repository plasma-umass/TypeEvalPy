# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [71, 54, 49]


def func2():
    return 26


def func3():
    return (96, 47, 8)


def func4():
    return {'unxof': 36, 'wjrnh': 83, 'kikwb': 94}


def func5():
    return 'jfrov'


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
