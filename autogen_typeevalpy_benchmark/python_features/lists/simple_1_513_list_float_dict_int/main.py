# Functions are assigned as elements of a list and then called.


def func1():
    return [24, 59, 47]


def func2():
    return 36.92


def func3():
    return {'qrdhx': 45, 'paqlg': 39, 'oheca': 97}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 74


b = ["Hello"]
b[0] = func4

f = b[0]()
