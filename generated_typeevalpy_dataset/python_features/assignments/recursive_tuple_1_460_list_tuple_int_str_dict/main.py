# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [98, 13, 56]


def func2():
    return (56, 72, 8)


def func3():
    return 44


def func4():
    return 'kkrnm'


def func5():
    return {'mlctn': 80, 'hrlsx': 85, 'bonfd': 45}


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
