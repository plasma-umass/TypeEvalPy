# Two variables are assigned a value via a tuple assignment.
def func1():
    return {'scijn': 28, 'jbjfc': 42, 'ealgt': 66}


def func2():
    return (58, 43, 7)


def func3():
    return [52, 93, 57]


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
