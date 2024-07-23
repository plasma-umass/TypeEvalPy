# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 'ezhjv'


def func2():
    return (33, 50, 91)


def func3():
    return {'xlwsp': 57, 'jldvb': 12, 'ljigg': 75}


def func4():
    return 64


(a, b), (c, d) = [(func1, func2), (func3, func4)]
