# Functions are assigned as elements of a list and then called.


def func1():
    return 'tpjgy'


def func2():
    return False


def func3():
    return 62.57


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'jnatb': 74, 'iglik': 13, 'kbzff': 39}


b = ["Hello"]
b[0] = func4

f = b[0]()
