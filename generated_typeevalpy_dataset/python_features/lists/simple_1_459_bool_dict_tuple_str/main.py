# Functions are assigned as elements of a list and then called.


def func1():
    return True


def func2():
    return {'yomba': 86, 'hkntl': 59, 'oonkh': 33}


def func3():
    return (40, 73, 92)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return 'znfop'


b = ["Hello"]
b[0] = func4

f = b[0]()
