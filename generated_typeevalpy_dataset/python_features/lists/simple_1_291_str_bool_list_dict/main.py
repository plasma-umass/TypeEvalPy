# Functions are assigned as elements of a list and then called.


def func1():
    return 'kveql'


def func2():
    return True


def func3():
    return [5, 100, 84]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'rnovg': 96, 'vmkam': 99, 'komfh': 74}


b = ["Hello"]
b[0] = func4

f = b[0]()
