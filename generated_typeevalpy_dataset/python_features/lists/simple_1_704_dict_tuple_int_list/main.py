# Functions are assigned as elements of a list and then called.


def func1():
    return {'kngwg': 4, 'kqsaa': 12, 'twerd': 98}


def func2():
    return (6, 31, 73)


def func3():
    return 72


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [39, 80, 21]


b = ["Hello"]
b[0] = func4

f = b[0]()
