# Functions are assigned to variables via starred assignment
def func1():
    return (62, 22, 87)


def func2():
    return 'nmjkm'


def func3():
    return [72, 16, 85]


def func4():
    return 27

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
