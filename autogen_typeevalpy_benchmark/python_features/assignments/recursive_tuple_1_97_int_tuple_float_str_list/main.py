# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 18


def func2():
    return (82, 78, 95)


def func3():
    return 60.27


def func4():
    return 'mknqq'


def func5():
    return [87, 73, 21]


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
