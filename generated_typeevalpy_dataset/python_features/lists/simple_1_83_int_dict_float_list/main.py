# Functions are assigned as elements of a list and then called.


def func1():
    return 91


def func2():
    return {'qyslh': 10, 'xvvhc': 46, 'ybicz': 65}


def func3():
    return 69.69


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [65, 80, 33]


b = ["Hello"]
b[0] = func4

f = b[0]()
