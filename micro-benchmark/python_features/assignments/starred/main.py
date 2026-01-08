# Functions are assigned to variables via starred assignment
def func1():
    return "Hello from func1"


def func2():
    return 42


def func3():
    return 42.5


def func4():
    return [2, 4]


a, *b, c = func1, func2, func3, func4; b_0=b[0]; b_1=b[1]

d = a()
e = b[0]()
f = b[1]()
g = c(); g_0=g[0]; g_1=g[1]
