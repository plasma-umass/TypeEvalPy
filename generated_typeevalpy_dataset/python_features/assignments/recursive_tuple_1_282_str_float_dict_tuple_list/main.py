# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'uhdnr'


def func2():
    return 56.87


def func3():
    return {'bbbaf': 68, 'tlsgc': 30, 'mpriu': 49}


def func4():
    return (46, 95, 84)


def func5():
    return [23, 71, 81]


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
