# Functions are assigned as elements of a list and then called.


def func1():
    return 42


def func2():
    return 42.5


def func3():
    return "Hello from func3"


a = [func1, func2, func3]; a_0=a[0]; a_1=a[1]; a_2=a[2]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return True


b = ["Hello"]; b_0=b[0]
b[0] = func4; b_0__2=b[0]

f = b[0]()
