# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'xzlam': 41, 'nexbu': 88, 'mmavl': 9}


def func2():
    return 'idrkv'


def func3():
    return 28


def func4():
    return 92.73


def func5():
    return (94, 29, 62)


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
