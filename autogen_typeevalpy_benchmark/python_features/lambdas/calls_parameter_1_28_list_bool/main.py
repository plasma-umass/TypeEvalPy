# A parameter is passed to a lambda and the lambda calls it.


def func1():
    return [35, 96, 61]


def func2():
    return True


x = lambda x: x()

a = x(func1)

b = x(func2)
