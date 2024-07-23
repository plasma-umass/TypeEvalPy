# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return {'kbnki': 100, 'tsbaf': 10, 'emxep': 29}


def func2():
    return [61, 50, 91]


def func3():
    return 20.57


def func4():
    return (13, 38, 99)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
