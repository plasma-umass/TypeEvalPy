# Functions are assigned as elements of a list and then called.


def func1():
    return [95, 33, 50]


def func2():
    return 'dwrlu'


def func3():
    return 10.07


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (17, 11, 25)


b = ["Hello"]
b[0] = func4

f = b[0]()
