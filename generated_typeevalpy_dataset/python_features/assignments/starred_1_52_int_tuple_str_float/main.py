# Functions are assigned to variables via starred assignment
def func1():
    return 3


def func2():
    return (100, 49, 71)


def func3():
    return 'ghiuq'


def func4():
    return 55.55

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
