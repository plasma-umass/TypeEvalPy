# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (71, 95, 61)


def func2():
    return 92.8


def func3():
    return 41


def func4():
    return [86, 9, 6]


def func5():
    return {'dkjrr': 55, 'gralb': 58, 'knkvm': 80}


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
