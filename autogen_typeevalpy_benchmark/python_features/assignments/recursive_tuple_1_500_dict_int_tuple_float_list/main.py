# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'ucrmx': 21, 'bgzhv': 21, 'bfuvy': 72}


def func2():
    return 79


def func3():
    return (36, 55, 51)


def func4():
    return 34.5


def func5():
    return [58, 97, 44]


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
