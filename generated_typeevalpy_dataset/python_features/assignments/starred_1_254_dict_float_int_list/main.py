# Functions are assigned to variables via starred assignment
def func1():
    return {'dggxa': 25, 'zifbu': 34, 'runio': 20}


def func2():
    return 29.47


def func3():
    return 62


def func4():
    return [98, 24, 83]

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
