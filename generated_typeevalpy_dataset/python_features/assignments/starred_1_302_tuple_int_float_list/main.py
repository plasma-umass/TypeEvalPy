# Functions are assigned to variables via starred assignment
def func1():
    return (98, 59, 35)


def func2():
    return 41


def func3():
    return 65.66


def func4():
    return [1, 61, 24]

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
