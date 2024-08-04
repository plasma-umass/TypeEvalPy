# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 'grepo'


def func2():
    return 83


def func3():
    return [87, 83, 52]


def func4():
    return (97, 43, 56)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
