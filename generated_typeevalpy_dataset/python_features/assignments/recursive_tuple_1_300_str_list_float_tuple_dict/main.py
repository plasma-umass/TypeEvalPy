# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'ehzio'


def func2():
    return [2, 27, 98]


def func3():
    return 32.16


def func4():
    return (38, 48, 8)


def func5():
    return {'ckpme': 62, 'dasyx': 51, 'ozjgi': 58}


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
