# Two variables are assigned a value via a tuple assignment.
def func1():
    return 90


def func2():
    return 'fzsav'


def func3():
    return 67.55


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
