# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'gvexl'


def func2():
    return [33, 72, 20]


def func3():
    return 62.63


def func4():
    return 88


def func5():
    return (18, 78, 23)


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
