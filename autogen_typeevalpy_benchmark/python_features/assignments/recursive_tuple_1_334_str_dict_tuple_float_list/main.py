# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 'jcgix'


def func2():
    return {'efugh': 39, 'voowv': 13, 'oprho': 52}


def func3():
    return (97, 6, 40)


def func4():
    return 91.96


def func5():
    return [54, 12, 85]


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
