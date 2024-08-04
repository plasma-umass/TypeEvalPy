# Functions are assigned as elements of a list and then called.


def func1():
    return (35, 75, 58)


def func2():
    return 20.16


def func3():
    return {'hkzec': 86, 'epslq': 18, 'himwt': 16}


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 87


b = ["Hello"]
b[0] = func4

f = b[0]()
