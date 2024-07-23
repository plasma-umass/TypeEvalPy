# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return (91, 47, 52)


def func2():
    return 76


def func3():
    return [59, 13, 62]


def func4():
    return {'rdkez': 74, 'lfryt': 74, 'xikaw': 100}


(a, b), (c, d) = [(func1, func2), (func3, func4)]
