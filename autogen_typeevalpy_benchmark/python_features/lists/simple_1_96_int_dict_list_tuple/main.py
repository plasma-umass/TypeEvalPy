# Functions are assigned as elements of a list and then called.


def func1():
    return 12


def func2():
    return {'dtzft': 98, 'ipcch': 96, 'igoeu': 25}


def func3():
    return [91, 88, 98]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (30, 7, 1)


b = ["Hello"]
b[0] = func4

f = b[0]()
