# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (13, 11, 95)


def func2():
    return 'jdfmv'


def func3():
    return 5


def func4():
    return 8.27


def func5():
    return {'ahyit': 92, 'rqsps': 33, 'ktzjg': 74}


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
