# A new list is created as a slice of another one containing functions.


def func1():
    return 10


def func2():
    return 75.52


def func3():
    return 'ppusq'


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
