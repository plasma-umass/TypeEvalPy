# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return 20


def func2():
    return 23.55


def func3():
    return {'lcgnf': 96, 'gqxaf': 21, 'tembu': 47}


def func4():
    return 'xacbn'


(a, b), (c, d) = [(func1, func2), (func3, func4)]
