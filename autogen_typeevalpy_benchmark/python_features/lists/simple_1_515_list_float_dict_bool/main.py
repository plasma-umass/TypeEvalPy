# Functions are assigned as elements of a list and then called.


def func1():
    return [35, 95, 86]


def func2():
    return 21.71


def func3():
    return {'jhwdm': 33, 'ndwhw': 11, 'avmnw': 1}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return False


b = ["Hello"]
b[0] = func4

f = b[0]()
