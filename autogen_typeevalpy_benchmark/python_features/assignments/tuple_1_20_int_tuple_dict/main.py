# Two variables are assigned a value via a tuple assignment.
def func1():
    return 19


def func2():
    return (20, 39, 85)


def func3():
    return {'uuwxb': 8, 'obevf': 80, 'qosik': 12}


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
