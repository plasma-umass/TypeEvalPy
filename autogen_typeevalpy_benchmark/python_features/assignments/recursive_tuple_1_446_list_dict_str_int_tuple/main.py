# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [90, 98, 46]


def func2():
    return {'lznow': 47, 'jhcwt': 43, 'zoyyu': 66}


def func3():
    return 'iylls'


def func4():
    return 86


def func5():
    return (96, 18, 63)


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
