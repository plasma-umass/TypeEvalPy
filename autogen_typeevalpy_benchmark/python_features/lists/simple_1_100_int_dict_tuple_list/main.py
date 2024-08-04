# Functions are assigned as elements of a list and then called.


def func1():
    return 20


def func2():
    return {'clwfb': 92, 'fqnlo': 100, 'eqnae': 87}


def func3():
    return (21, 76, 85)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [45, 43, 100]


b = ["Hello"]
b[0] = func4

f = b[0]()
