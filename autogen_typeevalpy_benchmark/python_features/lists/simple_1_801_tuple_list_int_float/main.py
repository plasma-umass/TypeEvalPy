# Functions are assigned as elements of a list and then called.


def func1():
    return (55, 85, 92)


def func2():
    return [90, 55, 15]


def func3():
    return 19


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 28.75


b = ["Hello"]
b[0] = func4

f = b[0]()
