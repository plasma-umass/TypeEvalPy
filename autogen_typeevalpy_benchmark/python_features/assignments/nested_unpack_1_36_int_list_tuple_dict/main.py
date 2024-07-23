# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 56


def func2():
    return [23, 68, 61]


def func3():
    return (92, 30, 89)


def func4():
    return {'zohxa': 61, 'tsvhn': 12, 'cheav': 75}


(a, b), (c, d) = [(func1, func2), (func3, func4)]
