# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [85, 84, 61]


def func2():
    return 42.26


def func3():
    return 'tyccf'


def func4():
    return {'cgxmq': 91, 'atzne': 83, 'wgfjt': 6}


def func5():
    return 53


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
