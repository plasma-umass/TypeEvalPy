# Functions are assigned as elements of a list and then called.


def func1():
    return 29.91


def func2():
    return 'eoddp'


def func3():
    return {'osine': 3, 'saidv': 84, 'sfvcd': 87}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 59


b = ["Hello"]
b[0] = func4

f = b[0]()
