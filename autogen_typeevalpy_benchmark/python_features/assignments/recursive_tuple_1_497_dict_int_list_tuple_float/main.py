# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return {'ahowj': 60, 'bsfba': 43, 'xwcuf': 51}


def func2():
    return 46


def func3():
    return [68, 91, 77]


def func4():
    return (51, 25, 43)


def func5():
    return 58.65


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
