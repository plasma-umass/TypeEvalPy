# Functions are assigned as elements of a list and then called.


def func1():
    return True


def func2():
    return [81, 51, 32]


def func3():
    return 95.73


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'yivmf'


b = ["Hello"]
b[0] = func4

f = b[0]()
