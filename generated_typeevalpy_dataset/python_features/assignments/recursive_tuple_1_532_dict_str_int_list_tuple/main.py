# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'ybxsv': 83, 'vvbfj': 60, 'ucicn': 71}


def func2():
    return 'cvova'


def func3():
    return 64


def func4():
    return [50, 46, 82]


def func5():
    return (69, 79, 18)


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
