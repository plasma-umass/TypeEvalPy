# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'wabas': 31, 'cyvpe': 80, 'uxuyw': 5}


def func2():
    return (36, 34, 70)


def func3():
    return 57.67


def func4():
    return 5


def func5():
    return [75, 29, 23]


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
