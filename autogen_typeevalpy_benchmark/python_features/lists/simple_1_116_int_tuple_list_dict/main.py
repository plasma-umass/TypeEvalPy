# Functions are assigned as elements of a list and then called.


def func1():
    return 87


def func2():
    return (75, 52, 48)


def func3():
    return [20, 100, 18]


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'dqjou': 48, 'swaof': 25, 'gkpzm': 79}


b = ["Hello"]
b[0] = func4

f = b[0]()
