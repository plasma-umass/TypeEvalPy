# Functions are assigned to variables via starred assignment
def func1():
    return 'xcfav'


def func2():
    return {'nghgo': 63, 'fmbze': 53, 'owzqu': 73}


def func3():
    return (43, 10, 89)


def func4():
    return [88, 58, 99]

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
