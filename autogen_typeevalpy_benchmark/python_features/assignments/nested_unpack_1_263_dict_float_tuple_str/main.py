# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return {'gpyyq': 98, 'zakun': 1, 'afmwi': 35}


def func2():
    return 68.84


def func3():
    return (7, 7, 31)


def func4():
    return 'osngv'


(a, b), (c, d) = [(func1, func2), (func3, func4)]
