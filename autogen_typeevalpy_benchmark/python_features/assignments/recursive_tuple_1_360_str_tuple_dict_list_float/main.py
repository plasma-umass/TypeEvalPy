# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'owxae'


def func2():
    return (4, 4, 76)


def func3():
    return {'cucdp': 54, 'ygjms': 1, 'ghzds': 80}


def func4():
    return [88, 28, 51]


def func5():
    return 37.42


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
