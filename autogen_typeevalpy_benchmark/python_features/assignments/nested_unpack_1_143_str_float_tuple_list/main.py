# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 'rddoi'


def func2():
    return 56.14


def func3():
    return (77, 59, 14)


def func4():
    return [100, 41, 61]


(a, b), (c, d) = [(func1, func2), (func3, func4)]
