# Functions are assigned as elements of a list and then called.


def func1():
    return 'yeghu'


def func2():
    return 47


def func3():
    return [41, 81, 2]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'gaooi': 11, 'umlgj': 53, 'rnoyi': 26}


b = ["Hello"]
b[0] = func4

f = b[0]()
