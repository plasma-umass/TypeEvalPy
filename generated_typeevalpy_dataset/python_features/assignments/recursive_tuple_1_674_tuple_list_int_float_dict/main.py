# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (49, 68, 67)


def func2():
    return [20, 83, 31]


def func3():
    return 42


def func4():
    return 78.96


def func5():
    return {'iaycj': 73, 'hwgxi': 52, 'eknjf': 28}


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
