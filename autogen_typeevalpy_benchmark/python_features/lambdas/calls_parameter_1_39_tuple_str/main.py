# A parameter is passed to a lambda and the lambda calls it.


def func1():
    return (100, 76, 78)


def func2():
    return 'kznvy'


x = lambda x: x()

a = x(func1)

b = x(func2)
