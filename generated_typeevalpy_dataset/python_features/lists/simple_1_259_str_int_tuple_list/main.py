# Functions are assigned as elements of a list and then called.


def func1():
    return 'fjfuo'


def func2():
    return 15


def func3():
    return (22, 97, 6)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [58, 98, 25]


b = ["Hello"]
b[0] = func4

f = b[0]()
