# Functions are assigned as elements of a list and then called.


def func1():
    return 'edten'


def func2():
    return 11.83


def func3():
    return True


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (98, 97, 61)


b = ["Hello"]
b[0] = func4

f = b[0]()
