# Two variables are assigned a value via a tuple assignment.
def func1():
    return 71


def func2():
    return [46, 17, 62]


def func3():
    return (19, 56, 21)


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
