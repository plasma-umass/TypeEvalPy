# A new list is created as a slice of another one containing functions.


def func1():
    return [12, 43, 92]


def func2():
    return (89, 94, 37)


def func3():
    return 'yxdrm'


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
