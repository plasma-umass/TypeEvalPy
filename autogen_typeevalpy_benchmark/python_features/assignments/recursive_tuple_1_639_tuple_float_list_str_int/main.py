# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (87, 85, 10)


def func2():
    return 78.62


def func3():
    return [91, 9, 95]


def func4():
    return 'rtzia'


def func5():
    return 30


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
