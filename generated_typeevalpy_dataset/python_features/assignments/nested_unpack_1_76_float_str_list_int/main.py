# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 47.3


def func2():
    return 'cuzus'


def func3():
    return [72, 34, 39]


def func4():
    return 43


(a, b), (c, d) = [(func1, func2), (func3, func4)]
