# Returning different types


# return_tuple
def func():
    return [18, 93, 84]


a = func()


# return_dict
def func1():
    return {'nsziv': 51, 'rzqvw': 26, 'xwpak': 51}


b = func1()

from collections import namedtuple


# return_namedtuple
def func3():
    Point = namedtuple("Point", ["x", "y"])
    return Point(1, 2)


c = func3()


# return_set
def func4():
    return 35


d = func4()


# return_list_comprehension
def func5():
    return [x**2 for x in range(1, 6)]


e = func5()
