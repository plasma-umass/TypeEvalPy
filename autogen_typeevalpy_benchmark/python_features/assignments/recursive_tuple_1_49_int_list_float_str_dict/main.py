# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 7


def func2():
    return [65, 74, 5]


def func3():
    return 30.99


def func4():
    return 'fggnb'


def func5():
    return {'dyyig': 24, 'kwsii': 96, 'iktlz': 90}


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
