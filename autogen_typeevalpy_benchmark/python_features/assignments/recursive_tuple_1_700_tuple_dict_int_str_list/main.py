# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (19, 12, 68)


def func2():
    return {'unjdq': 55, 'rkmot': 38, 'ncrqn': 65}


def func3():
    return 18


def func4():
    return 'upsve'


def func5():
    return [30, 23, 17]


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
