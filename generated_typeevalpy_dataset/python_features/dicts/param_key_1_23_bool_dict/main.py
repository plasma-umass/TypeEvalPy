# The key of a dictionary is passed as a function parameter.


def func1(key="a"):
    return d[key]()


def func2():
    return True


def func3():
    return {'eraih': 40, 'opddq': 78, 'ksave': 14}


d = {"a": func2, "b": func3}

e = func1()
f = func1("b")
