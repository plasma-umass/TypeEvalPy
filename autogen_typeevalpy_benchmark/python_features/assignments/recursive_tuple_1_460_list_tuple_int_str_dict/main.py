# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [97, 58, 11]


def func2():
    return (73, 86, 15)


def func3():
    return 4


def func4():
    return 'jpvsr'


def func5():
    return {'qpcua': 34, 'wyewz': 51, 'gmyon': 58}


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
