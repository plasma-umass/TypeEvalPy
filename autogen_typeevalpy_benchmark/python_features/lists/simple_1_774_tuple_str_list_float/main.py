# Functions are assigned as elements of a list and then called.


def func1():
    return (88, 35, 86)


def func2():
    return 'xvfqu'


def func3():
    return [90, 11, 94]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 79.49


b = ["Hello"]
b[0] = func4

f = b[0]()
