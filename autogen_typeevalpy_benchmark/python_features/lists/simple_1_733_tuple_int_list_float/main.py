# Functions are assigned as elements of a list and then called.


def func1():
    return (99, 28, 77)


def func2():
    return 45


def func3():
    return [13, 75, 90]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 52.71


b = ["Hello"]
b[0] = func4

f = b[0]()
