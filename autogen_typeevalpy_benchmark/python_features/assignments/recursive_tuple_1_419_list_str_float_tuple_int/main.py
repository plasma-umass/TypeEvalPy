# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [75, 7, 7]


def func2():
    return 'whofd'


def func3():
    return 6.95


def func4():
    return (100, 73, 97)


def func5():
    return 18


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
