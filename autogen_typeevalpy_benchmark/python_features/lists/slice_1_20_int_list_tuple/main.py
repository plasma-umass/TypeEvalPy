# A new list is created as a slice of another one containing functions.


def func1():
    return 58


def func2():
    return [10, 86, 93]


def func3():
    return (18, 11, 46)


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
