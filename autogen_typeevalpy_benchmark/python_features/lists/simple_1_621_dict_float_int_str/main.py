# Functions are assigned as elements of a list and then called.


def func1():
    return {'fxfiy': 10, 'jlcsm': 29, 'sukoy': 12}


def func2():
    return 10.16


def func3():
    return 21


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'isyyp'


b = ["Hello"]
b[0] = func4

f = b[0]()
