# Three variables are assigned a value via a recursive tuple assignment


def func1():
    return 88


def func2():
    return 64.01


def func3():
    return 'gyzaf'


def func4():
    return {'syxoa': 78, 'hjoqu': 55, 'czkyc': 11}


def func5():
    return (92, 59, 6)


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
