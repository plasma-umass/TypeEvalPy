# Functions are assigned as elements of a list and then called.


def func1():
    return 43.7


def func2():
    return {'xuwgn': 35, 'araqp': 55, 'nkmud': 41}


def func3():
    return False


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (26, 1, 79)


b = ["Hello"]
b[0] = func4

f = b[0]()
