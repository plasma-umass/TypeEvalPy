# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (83, 91, 4)


def func2():
    return {'rfour': 41, 'cvqxl': 43, 'yxvko': 71}


def func3():
    return 32.99


def func4():
    return [45, 64, 48]


def func5():
    return 76


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
