# Functions are assigned to variables via starred assignment
def func1():
    return [64, 68, 81]


def func2():
    return 92.62


def func3():
    return (54, 26, 22)


def func4():
    return {'rywey': 99, 'pzxkz': 7, 'caegs': 47}

a, *b, c = func1, func2, func3, func4

d = a()
e = b[0]()
f = b[1]()
g = c()
