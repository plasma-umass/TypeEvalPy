# Functions are assigned as elements of a list and then called.


def func1():
    return [72, 56, 18]


def func2():
    return (85, 86, 14)


def func3():
    return 'bhaju'


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'ddgnq': 12, 'oeitl': 92, 'qhlba': 49}


b = ["Hello"]
b[0] = func4

f = b[0]()
