# Two variables are assigned a value via a tuple assignment.
def func1():
    return (10, 2, 77)


def func2():
    return 'txzco'


def func3():
    return 46.12


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
