# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 9


def func2():
    return 48.93


def func3():
    return {'kgxgz': 48, 'rgouh': 70, 'kcqfz': 22}


def func4():
    return (32, 69, 55)


def func5():
    return [72, 58, 6]


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
