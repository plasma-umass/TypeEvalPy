# Functions are assigned as elements of a list and then called.


def func1():
    return 53.79


def func2():
    return 57


def func3():
    return {'lvyuj': 95, 'pdogw': 71, 'wuybs': 55}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return True


b = ["Hello"]
b[0] = func4

f = b[0]()
