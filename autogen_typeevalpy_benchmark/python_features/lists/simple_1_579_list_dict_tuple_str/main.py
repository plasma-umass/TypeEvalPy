# Functions are assigned as elements of a list and then called.


def func1():
    return [43, 31, 6]


def func2():
    return {'txmrk': 100, 'fcttb': 38, 'jnbgu': 8}


def func3():
    return (50, 18, 12)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'tobzl'


b = ["Hello"]
b[0] = func4

f = b[0]()
