# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 56.9


def func2():
    return 4


def func3():
    return [24, 33, 45]


def func4():
    return 'zjbqb'


def func5():
    return (75, 15, 68)


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
