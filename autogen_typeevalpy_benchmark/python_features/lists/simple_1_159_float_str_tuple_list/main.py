# Functions are assigned as elements of a list and then called.


def func1():
    return 38.75


def func2():
    return 'yxxfd'


def func3():
    return (88, 86, 68)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [84, 25, 78]


b = ["Hello"]
b[0] = func4

f = b[0]()
