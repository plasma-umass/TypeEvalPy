# Functions are assigned as elements of a list and then called.


def func1():
    return False


def func2():
    return [99, 9, 28]


def func3():
    return 47


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'ydrys': 63, 'nodfw': 62, 'eqyoz': 93}


b = ["Hello"]
b[0] = func4

f = b[0]()
