# Functions are assigned as elements of a list and then called.


def func1():
    return (49, 74, 46)


def func2():
    return 51.75


def func3():
    return [66, 88, 59]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'arqjw'


b = ["Hello"]
b[0] = func4

f = b[0]()
