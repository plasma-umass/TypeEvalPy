# Functions are assigned as elements of a list and then called.


def func1():
    return {'vrlbo': 22, 'dtwoi': 91, 'ofveg': 49}


def func2():
    return True


def func3():
    return 'yjdtn'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return [38, 75, 50]


b = ["Hello"]
b[0] = func4

f = b[0]()
