# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return [10, 65, 38]


def func2():
    return {'idngn': 4, 'ugljm': 62, 'giurx': 15}


def func3():
    return (34, 9, 49)


def func4():
    return 82


(a, b), (c, d) = [(func1, func2), (func3, func4)]
