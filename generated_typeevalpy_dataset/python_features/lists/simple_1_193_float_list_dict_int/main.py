# Functions are assigned as elements of a list and then called.


def func1():
    return 26.51


def func2():
    return [70, 90, 39]


def func3():
    return {'injfd': 54, 'dlhgl': 14, 'boskv': 18}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 76


b = ["Hello"]
b[0] = func4

f = b[0]()
