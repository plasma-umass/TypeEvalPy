# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 3.14


def func2():
    return {'gwban': 19, 'cekrs': 89, 'acmns': 12}


def func3():
    return 28


def func4():
    return (82, 60, 70)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
