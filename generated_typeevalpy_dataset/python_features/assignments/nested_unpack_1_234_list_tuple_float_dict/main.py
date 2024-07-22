# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return [59, 75, 84]


def func2():
    return (52, 15, 79)


def func3():
    return 46.11


def func4():
    return {'zbwqq': 5, 'rnuhh': 42, 'yglvc': 2}


(a, b), (c, d) = [(func1, func2), (func3, func4)]
