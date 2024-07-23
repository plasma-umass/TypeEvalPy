# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 57.95


def func2():
    return (14, 57, 79)


def func3():
    return [56, 83, 37]


def func4():
    return 'ayxin'


(a, b), (c, d) = [(func1, func2), (func3, func4)]
