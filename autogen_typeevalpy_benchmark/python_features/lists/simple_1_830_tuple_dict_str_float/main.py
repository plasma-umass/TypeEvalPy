# Functions are assigned as elements of a list and then called.


def func1():
    return (19, 40, 84)


def func2():
    return {'vhhzg': 70, 'avoyu': 100, 'kmdkd': 41}


def func3():
    return 'zckxc'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 97.95


b = ["Hello"]
b[0] = func4

f = b[0]()
