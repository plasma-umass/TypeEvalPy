# Functions are assigned as elements of a list and then called.


def func1():
    return [80, 82, 81]


def func2():
    return 'yhebu'


def func3():
    return True


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 80.56


b = ["Hello"]
b[0] = func4

f = b[0]()
