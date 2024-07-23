# example of nested unpacking in Python. The list [(func1, func2), (func3, func4)] contains two tuples, each with two function objects.
# The outer parentheses in the assignment (a, b), (c, d) = ... are used for nested unpacking.


def func1():
    return {'kycvm': 25, 'gzbap': 6, 'chsud': 63}


def func2():
    return 'wodzt'


def func3():
    return 13.71


def func4():
    return [24, 64, 32]


(a, b), (c, d) = [(func1, func2), (func3, func4)]
