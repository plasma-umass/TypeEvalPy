# Functions are assigned as elements of a list and then called.


def func1():
    return 'ljfeu'


def func2():
    return (36, 51, 42)


def func3():
    return 10


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [70, 25, 88]


b = ["Hello"]
b[0] = func4

f = b[0]()
