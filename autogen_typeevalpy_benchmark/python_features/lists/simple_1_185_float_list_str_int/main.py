# Functions are assigned as elements of a list and then called.


def func1():
    return 22.83


def func2():
    return [18, 77, 46]


def func3():
    return 'rqstn'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 58


b = ["Hello"]
b[0] = func4

f = b[0]()
