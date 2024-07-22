# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return {'ffyer': 97, 'etqum': 63, 'tdinu': 3}


def func2():
    return 29


def func3():
    return [75, 41, 70]


def func4():
    return 40.25


(a, b), (c, d) = [(func1, func2), (func3, func4)]
