# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [28, 41, 13]


def func2():
    return {'shlux': 42, 'dcdoq': 45, 'rssav': 60}


def func3():
    return 84


def func4():
    return (8, 44, 47)


def func5():
    return 81.48


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
