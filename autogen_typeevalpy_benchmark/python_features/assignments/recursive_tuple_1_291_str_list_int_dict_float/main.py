# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'wbdxe'


def func2():
    return [93, 64, 41]


def func3():
    return 51


def func4():
    return {'yrddn': 69, 'komle': 91, 'vkatx': 57}


def func5():
    return 64.47


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
