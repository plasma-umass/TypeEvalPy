# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 48.06


def func2():
    return (61, 56, 71)


def func3():
    return {'yhedc': 88, 'dvgku': 90, 'vryub': 52}


def func4():
    return 24


(a, b), (c, d) = [(func1, func2), (func3, func4)]
