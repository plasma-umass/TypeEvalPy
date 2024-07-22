# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 'sazuk'


def func2():
    return 36.58


def func3():
    return {'escik': 96, 'ucsiy': 14, 'lgcbe': 6}


def func4():
    return [80, 67, 65]


(a, b), (c, d) = [(func1, func2), (func3, func4)]
