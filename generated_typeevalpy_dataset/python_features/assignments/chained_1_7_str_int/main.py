# Two variables are assigned a function via chained assignment.


def func1():
    return 'ilxfs'


def func2():
    return 74


a = b = func1

c = b()

a = b = func2

d = a()
