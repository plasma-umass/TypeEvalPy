# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return (50, 56, 26)


def func2():
    return 27


def func3():
    return 'ceway'


def func4():
    return [26, 24, 20]


def func5():
    return {'ymwxe': 80, 'jauyd': 75, 'yqowf': 48}


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
