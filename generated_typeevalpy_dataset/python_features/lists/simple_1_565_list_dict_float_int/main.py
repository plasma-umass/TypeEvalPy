# Functions are assigned as elements of a list and then called.


def func1():
    return [74, 2, 70]


def func2():
    return {'efsor': 36, 'wljam': 70, 'kqclq': 21}


def func3():
    return 96.57


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 50


b = ["Hello"]
b[0] = func4

f = b[0]()
