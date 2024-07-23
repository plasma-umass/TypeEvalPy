# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return [55, 57, 86]


def func2():
    return {'ezujz': 85, 'rzqjk': 77, 'dlduv': 21}


def func3():
    return 95


def func4():
    return (14, 61, 7)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
