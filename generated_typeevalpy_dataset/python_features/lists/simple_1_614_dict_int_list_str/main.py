# Functions are assigned as elements of a list and then called.


def func1():
    return {'lpkpk': 6, 'fbzmw': 7, 'nynhu': 48}


def func2():
    return 89


def func3():
    return [46, 72, 76]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'bilpg'


b = ["Hello"]
b[0] = func4

f = b[0]()
