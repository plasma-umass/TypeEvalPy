# A parameter is passed to a lambda and the lambda calls it.


def func1():
    return 42


def func2():
    return "Hello from func2"


x = lambda x: x()

a = x(func1)

b = x(func2)
