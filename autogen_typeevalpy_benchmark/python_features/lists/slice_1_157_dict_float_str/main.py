# A new list is created as a slice of another one containing functions.


def func1():
    return {'kjkto': 28, 'kuuot': 41, 'nproe': 61}


def func2():
    return 39.6


def func3():
    return 'rwnhi'


ls = [func1, func2, func3]

ls2 = ls[1:3]

c = ls2[0]()
