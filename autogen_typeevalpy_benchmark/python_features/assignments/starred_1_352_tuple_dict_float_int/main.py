# Functions are assigned to variables via starred assignment
def func1():
    return (10, 61, 8)


def func2():
    return {'nfbtc': 60, 'rtyyj': 27, 'jrzjk': 93}


def func3():
    return 70.74


def func4():
    return 24

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
