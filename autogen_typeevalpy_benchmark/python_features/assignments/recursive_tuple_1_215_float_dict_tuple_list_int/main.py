# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 58.88


def func2():
    return {'wxzld': 38, 'jhbvm': 22, 'celal': 36}


def func3():
    return (64, 50, 68)


def func4():
    return [94, 64, 61]


def func5():
    return 38


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
