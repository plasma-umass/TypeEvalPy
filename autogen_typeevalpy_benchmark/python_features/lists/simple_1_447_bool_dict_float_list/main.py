# Functions are assigned as elements of a list and then called.


def func1():
    return False


def func2():
    return {'fupkm': 30, 'sjxhw': 35, 'cucyj': 5}


def func3():
    return 58.51


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [14, 82, 74]


b = ["Hello"]
b[0] = func4

f = b[0]()
