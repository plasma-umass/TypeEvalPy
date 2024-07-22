# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return (69, 23, 83)


def func2():
    return 98


def func3():
    return {'hckqg': 48, 'anwrp': 64, 'blbel': 24}


def func4():
    return [92, 4, 11]


(a, b), (c, d) = [(func1, func2), (func3, func4)]
