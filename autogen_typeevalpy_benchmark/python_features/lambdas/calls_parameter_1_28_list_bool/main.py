# A parameter is passed to a lambda and the lambda calls it.


def func1():
    return [49, 54, 3]


def func2():
    return False


x = lambda x: x()

a = x(func1)

b = x(func2)
