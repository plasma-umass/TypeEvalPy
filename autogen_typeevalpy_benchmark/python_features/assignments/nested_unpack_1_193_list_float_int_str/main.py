# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return [92, 69, 75]


def func2():
    return 75.47


def func3():
    return 97


def func4():
    return 'bsvro'


(a, b), (c, d) = [(func1, func2), (func3, func4)]
