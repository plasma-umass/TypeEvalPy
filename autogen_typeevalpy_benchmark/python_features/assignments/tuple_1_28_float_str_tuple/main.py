# Two variables are assigned a value via a tuple assignment.
def func1():
    return 91.05


def func2():
    return 'jmspt'


def func3():
    return (65, 75, 10)


a, b = func1, func2
f = a()
g = b()

c, d, e = func1, func2, func3

h = e()
