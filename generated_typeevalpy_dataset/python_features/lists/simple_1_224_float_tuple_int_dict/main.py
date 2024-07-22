# Functions are assigned as elements of a list and then called.


def func1():
    return 49.79


def func2():
    return (65, 46, 42)


def func3():
    return 26


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'bjyou': 13, 'cswyt': 19, 'cwifp': 16}


b = ["Hello"]
b[0] = func4

f = b[0]()
