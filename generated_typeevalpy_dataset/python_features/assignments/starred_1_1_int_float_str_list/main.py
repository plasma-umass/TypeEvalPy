# Functions are assigned to variables via starred assignment
def func1():
    return 76


def func2():
    return 83.35


def func3():
    return 'hrzpe'


def func4():
    return [64, 2, 90]

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
