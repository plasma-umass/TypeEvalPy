# Functions are assigned as elements of a list and then called.


def func1():
    return 82.6


def func2():
    return (54, 5, 29)


def func3():
    return [8, 22, 42]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'biqjt'


b = ["Hello"]
b[0] = func4

f = b[0]()
