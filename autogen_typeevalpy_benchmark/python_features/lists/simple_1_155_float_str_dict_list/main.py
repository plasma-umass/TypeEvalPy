# Functions are assigned as elements of a list and then called.


def func1():
    return 13.73


def func2():
    return 'zeinc'


def func3():
    return {'qsqdt': 90, 'mcjca': 3, 'zwigm': 8}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [33, 85, 18]


b = ["Hello"]
b[0] = func4

f = b[0]()
