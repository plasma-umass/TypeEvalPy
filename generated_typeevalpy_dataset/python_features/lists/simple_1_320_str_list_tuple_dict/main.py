# Functions are assigned as elements of a list and then called.


def func1():
    return 'lwfnw'


def func2():
    return [14, 25, 84]


def func3():
    return (70, 78, 24)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'unibw': 33, 'kutjp': 88, 'ydzbm': 48}


b = ["Hello"]
b[0] = func4

f = b[0]()
