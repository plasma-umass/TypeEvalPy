# Functions are assigned to variables via starred assignment
def func1():
    return 2.5


def func2():
    return 'miekz'


def func3():
    return (54, 90, 31)


def func4():
    return [24, 56, 22]

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
