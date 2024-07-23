# A new list is created as a slice of another one containing functions.


def func1():
    return True


def func2():
    return 'vbltv'


def func3():
    return [56, 42, 57]


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
