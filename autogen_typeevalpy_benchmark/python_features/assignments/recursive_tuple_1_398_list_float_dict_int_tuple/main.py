# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [28, 26, 57]


def func2():
    return 78.93


def func3():
    return {'syrzy': 42, 'yieoo': 85, 'ptwyt': 75}


def func4():
    return 10


def func5():
    return (82, 39, 31)


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
