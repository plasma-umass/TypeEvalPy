# Functions are assigned as elements of a list and then called.


def func1():
    return 'xdzuj'


def func2():
    return (6, 38, 23)


def func3():
    return False


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'jubww': 10, 'ltrjs': 57, 'qpvqo': 57}


b = ["Hello"]
b[0] = func4

f = b[0]()
