# Two variables are assigned a value via a tuple assignment.
def func1():
    return 19


def func2():
    return 46.65


def func3():
    return {'fhokl': 2, 'zjvom': 11, 'ouxaz': 34}


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
