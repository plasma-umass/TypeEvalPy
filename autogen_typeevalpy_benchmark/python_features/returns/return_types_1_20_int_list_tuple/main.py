# Returning different types


# return_tuple
def func():
    return 27


a = func()


# return_dict
def func1():
    return [75, 13, 86]


b = func1()

from collections import namedtuple


# return_namedtuple
def func3():
    Point = namedtuple("Point", ["x", "y"])
    return Point(1, 2)


c = func3()


# return_set
def func4():
    return (47, 68, 42)


d = func4()


# return_list_comprehension
def func5():
    return [x**2 for x in range(1, 6)]


e = func5()
