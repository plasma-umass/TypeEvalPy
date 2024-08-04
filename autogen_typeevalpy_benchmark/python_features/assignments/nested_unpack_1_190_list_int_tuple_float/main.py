# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return [43, 36, 7]


def func2():
    return 65


def func3():
    return (29, 71, 83)


def func4():
    return 38.49


(a, b), (c, d) = [(func1, func2), (func3, func4)]
