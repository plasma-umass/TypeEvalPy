# Functions are assigned as elements of a list and then called.


def func1():
    return 'quhcc'


def func2():
    return 35.37


def func3():
    return {'wmpdf': 31, 'wqtvm': 43, 'ajrqd': 98}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (68, 100, 100)


b = ["Hello"]
b[0] = func4

f = b[0]()
