# Functions are assigned as elements of a list and then called.


def func1():
    return {'ywnpc': 16, 'bgoay': 26, 'kijje': 18}


def func2():
    return (99, 53, 17)


def func3():
    return [90, 7, 72]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 60


b = ["Hello"]
b[0] = func4

f = b[0]()
