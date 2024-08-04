# Functions are assigned as elements of a list and then called.


def func1():
    return 16.94


def func2():
    return False


def func3():
    return 'touba'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (66, 20, 11)


b = ["Hello"]
b[0] = func4

f = b[0]()
