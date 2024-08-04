# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 99


def func2():
    return 'jpzsu'


def func3():
    return {'eipyo': 16, 'xjyfp': 73, 'vgshr': 6}


def func4():
    return (45, 69, 55)


(a, b), (c, d) = [(func1, func2), (func3, func4)]
