# Functions are assigned to variables via starred assignment
def func1():
    return [7, 1, 11]


def func2():
    return 17


def func3():
    return 72.81


def func4():
    return 'scfml'

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
