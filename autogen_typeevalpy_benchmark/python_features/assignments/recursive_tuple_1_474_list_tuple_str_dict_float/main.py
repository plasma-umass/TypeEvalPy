# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return [18, 56, 16]


def func2():
    return (12, 38, 19)


def func3():
    return 'gurxt'


def func4():
    return {'yoirc': 53, 'nnxbt': 19, 'yprcq': 82}


def func5():
    return 24.77


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
