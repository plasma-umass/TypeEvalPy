# Functions are assigned as elements of a list and then called.


def func1():
    return (87, 75, 30)


def func2():
    return {'nppxf': 93, 'riihk': 87, 'yqjvx': 7}


def func3():
    return True


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'elkmw'


b = ["Hello"]
b[0] = func4

f = b[0]()
