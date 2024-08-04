# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return [99, 58, 59]


def func2():
    return 66.19


def func3():
    return 65


def func4():
    return (18, 64, 1)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
