# Functions are assigned as elements of a list and then called.


def func1():
    return 'upbqy'


def func2():
    return {'wofja': 95, 'uyrkc': 25, 'jlqzz': 51}


def func3():
    return 53.97


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 58


b = ["Hello"]
b[0] = func4

f = b[0]()
