# A new list is created as a slice of another one containing functions.


def func1():
    return 'mbaar'


def func2():
    return [23, 15, 70]


def func3():
    return 68.04


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
