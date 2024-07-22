# Functions are assigned as elements of a list and then called.


def func1():
    return {'rykju': 93, 'qsqfj': 29, 'woygv': 73}


def func2():
    return 20.69


def func3():
    return [2, 64, 83]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (99, 7, 26)


b = ["Hello"]
b[0] = func4

f = b[0]()
