# Functions are assigned as elements of a list and then called.


def func1():
    return {'nxgcf': 72, 'miuyy': 92, 'maznf': 7}


def func2():
    return 'ffvoa'


def func3():
    return (40, 95, 17)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 35


b = ["Hello"]
b[0] = func4

f = b[0]()
