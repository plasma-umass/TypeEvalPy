# Functions are assigned as elements of a list and then called.


def func1():
    return (96, 49, 61)


def func2():
    return 22.17


def func3():
    return 46


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'rgfuu'


b = ["Hello"]
b[0] = func4

f = b[0]()
