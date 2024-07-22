# A new list is created as a slice of another one containing functions.


def func1():
    return 38


def func2():
    return 52.81


def func3():
    return 'zxsvl'


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
