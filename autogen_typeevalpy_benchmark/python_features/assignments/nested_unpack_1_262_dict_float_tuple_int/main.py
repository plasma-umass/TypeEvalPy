# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return {'pvsky': 69, 'onqsb': 51, 'clwwg': 95}


def func2():
    return 4.89


def func3():
    return (50, 67, 98)


def func4():
    return 81


(a, b), (c, d) = [(func1, func2), (func3, func4)]
