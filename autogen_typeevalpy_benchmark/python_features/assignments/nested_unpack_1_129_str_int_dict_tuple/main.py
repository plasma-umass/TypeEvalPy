# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 'llafh'


def func2():
    return 91


def func3():
    return {'vbcwe': 59, 'ilgwn': 80, 'qpppj': 23}


def func4():
    return (87, 91, 39)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
