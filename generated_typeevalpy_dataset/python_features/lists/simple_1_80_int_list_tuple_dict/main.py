# Functions are assigned as elements of a list and then called.


def func1():
    return 70


def func2():
    return [17, 44, 2]


def func3():
    return (56, 36, 4)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'ktifl': 46, 'scgdx': 65, 'zyfuk': 75}


b = ["Hello"]
b[0] = func4

f = b[0]()
