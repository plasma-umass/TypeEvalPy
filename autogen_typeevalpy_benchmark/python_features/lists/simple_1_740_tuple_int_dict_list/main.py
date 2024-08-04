# Functions are assigned as elements of a list and then called.


def func1():
    return (43, 16, 68)


def func2():
    return 55


def func3():
    return {'hanti': 49, 'ecuzt': 46, 'ssvdi': 7}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [28, 61, 13]


b = ["Hello"]
b[0] = func4

f = b[0]()
