# A new list is created as a slice of another one containing functions.


def func1():
    return (88, 93, 18)


def func2():
    return 51


def func3():
    return 'uappd'


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
