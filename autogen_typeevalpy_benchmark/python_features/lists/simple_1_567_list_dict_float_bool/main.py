# Functions are assigned as elements of a list and then called.


def func1():
    return [70, 42, 66]


def func2():
    return {'fqpix': 37, 'azfho': 63, 'dqppj': 7}


def func3():
    return 34.45


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return False


b = ["Hello"]
b[0] = func4

f = b[0]()
