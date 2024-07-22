# Functions are assigned as elements of a list and then called.


def func1():
    return 92.46


def func2():
    return True


def func3():
    return {'mwvav': 81, 'frdsz': 80, 'dxyna': 36}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return (64, 88, 63)


b = ["Hello"]
b[0] = func4

f = b[0]()
