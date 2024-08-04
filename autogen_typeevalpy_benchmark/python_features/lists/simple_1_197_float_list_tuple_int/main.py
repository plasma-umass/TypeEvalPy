# Functions are assigned as elements of a list and then called.


def func1():
    return 17.11


def func2():
    return [51, 95, 95]


def func3():
    return (84, 100, 100)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 71


b = ["Hello"]
b[0] = func4

f = b[0]()
