# Functions are assigned as elements of a list and then called.


def func1():
    return 49.58


def func2():
    return [12, 2, 7]


def func3():
    return (7, 3, 45)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'iywcm'


b = ["Hello"]
b[0] = func4

f = b[0]()
