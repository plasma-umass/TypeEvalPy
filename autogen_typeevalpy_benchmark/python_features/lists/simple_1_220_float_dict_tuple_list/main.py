# Functions are assigned as elements of a list and then called.


def func1():
    return 35.69


def func2():
    return {'pjbob': 44, 'wcupw': 34, 'bkenn': 55}


def func3():
    return (60, 63, 93)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [86, 91, 22]


b = ["Hello"]
b[0] = func4

f = b[0]()
