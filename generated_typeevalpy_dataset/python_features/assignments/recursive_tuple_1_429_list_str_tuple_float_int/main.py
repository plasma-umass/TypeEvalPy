# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [18, 53, 38]


def func2():
    return 'ikbdz'


def func3():
    return (19, 4, 82)


def func4():
    return 27.18


def func5():
    return 19


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
