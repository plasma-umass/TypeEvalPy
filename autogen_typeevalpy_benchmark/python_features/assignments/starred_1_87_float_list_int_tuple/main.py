# Functions are assigned to variables via starred assignment
def func1():
    return 17.65


def func2():
    return [72, 97, 3]


def func3():
    return 99


def func4():
    return (84, 46, 81)

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
