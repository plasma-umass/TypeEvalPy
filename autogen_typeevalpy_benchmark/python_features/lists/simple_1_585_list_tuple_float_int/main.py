# Functions are assigned as elements of a list and then called.


def func1():
    return [57, 16, 82]


def func2():
    return (65, 4, 34)


def func3():
    return 36.75


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 79


b = ["Hello"]
b[0] = func4

f = b[0]()
