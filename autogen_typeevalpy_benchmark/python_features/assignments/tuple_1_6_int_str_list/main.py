# Two variables are assigned a value via a tuple assignment.
def func1():
    return 18


def func2():
    return 'xavpf'


def func3():
    return [95, 47, 84]


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
