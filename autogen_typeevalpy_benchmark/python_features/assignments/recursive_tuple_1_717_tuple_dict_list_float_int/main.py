# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (43, 83, 48)


def func2():
    return {'yjkid': 61, 'bldro': 51, 'tfwlz': 65}


def func3():
    return [19, 79, 86]


def func4():
    return 33.67


def func5():
    return 74


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
