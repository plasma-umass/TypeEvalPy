# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 'yijmw'


def func2():
    return 51


def func3():
    return (69, 17, 58)


def func4():
    return 87.14


(a, b), (c, d) = [(func1, func2), (func3, func4)]
