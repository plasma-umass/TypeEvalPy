# Functions are assigned as elements of a list and then called.


def func1():
    return [3, 51, 12]


def func2():
    return (43, 21, 75)


def func3():
    return True


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'duxry': 68, 'iaebk': 90, 'pzpxf': 94}


b = ["Hello"]
b[0] = func4

f = b[0]()
