# Functions are assigned as elements of a list and then called.


def func1():
    return (76, 20, 62)


def func2():
    return 'gcdyt'


def func3():
    return True


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 30


b = ["Hello"]
b[0] = func4

f = b[0]()
