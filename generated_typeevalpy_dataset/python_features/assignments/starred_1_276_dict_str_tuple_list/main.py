# Functions are assigned to variables via starred assignment
def func1():
    return {'gcohn': 73, 'fgadf': 83, 'kblkl': 7}


def func2():
    return 'odrjw'


def func3():
    return (37, 85, 26)


def func4():
    return [27, 24, 79]

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
