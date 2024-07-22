# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 73.4


def func2():
    return (21, 69, 30)


def func3():
    return 94


def func4():
    return [27, 58, 62]


(a, b), (c, d) = [(func1, func2), (func3, func4)]
