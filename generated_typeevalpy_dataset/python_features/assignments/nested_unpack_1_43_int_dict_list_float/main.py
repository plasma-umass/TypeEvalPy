# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 66


def func2():
    return {'avscc': 71, 'mdehl': 21, 'ovyhq': 22}


def func3():
    return [63, 95, 91]


def func4():
    return 49.18


(a, b), (c, d) = [(func1, func2), (func3, func4)]
