# Functions are assigned to variables via starred assignment
def func1():
    return [77, 70, 89]


def func2():
    return 'xmizq'


def func3():
    return (83, 97, 75)


def func4():
    return 51.11

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
