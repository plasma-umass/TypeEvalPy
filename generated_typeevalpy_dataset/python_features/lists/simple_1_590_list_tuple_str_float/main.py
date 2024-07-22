# Functions are assigned as elements of a list and then called.


def func1():
    return [44, 39, 92]


def func2():
    return (37, 59, 27)


def func3():
    return 'wcdxy'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 2.95


b = ["Hello"]
b[0] = func4

f = b[0]()
