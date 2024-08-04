# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return (15, 44, 22)


def func2():
    return 'ailwo'


def func3():
    return {'xgspz': 79, 'jkovc': 39, 'rinti': 47}


def func4():
    return 56


(a, b), (c, d) = [(func1, func2), (func3, func4)]
