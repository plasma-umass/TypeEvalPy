# A new list is created as a slice of another one containing functions.


def func1():
    return {'ufrhp': 50, 'dumrk': 44, 'mgizx': 37}


def func2():
    return 48


def func3():
    return (80, 94, 93)


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
