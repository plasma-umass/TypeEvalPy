# A new list is created as a slice of another one containing functions.


def func1():
    return (50, 59, 92)


def func2():
    return [67, 1, 3]


def func3():
    return 16.72


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
