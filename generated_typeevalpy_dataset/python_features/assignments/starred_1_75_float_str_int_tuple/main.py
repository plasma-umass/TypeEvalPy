# Functions are assigned to variables via starred assignment
def func1():
    return 3.03


def func2():
    return 'slrqu'


def func3():
    return 11


def func4():
    return (66, 25, 11)

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
