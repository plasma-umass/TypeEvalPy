# Functions are assigned as elements of a list and then called.


def func1():
    return 39


def func2():
    return False


def func3():
    return 'aposi'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (28, 78, 35)


b = ["Hello"]
b[0] = func4

f = b[0]()
