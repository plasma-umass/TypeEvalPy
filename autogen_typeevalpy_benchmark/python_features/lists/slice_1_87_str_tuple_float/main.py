# A new list is created as a slice of another one containing functions.


def func1():
    return 'ozhrz'


def func2():
    return (62, 74, 92)


def func3():
    return 50.88


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
