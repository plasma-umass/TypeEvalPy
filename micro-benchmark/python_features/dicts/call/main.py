# A dictionary containing functions as values is created.


def func1():
    return "Hello from func1"


def func2():
    return 42


d = {"a": func1, 1: func2, 2: 3}; d_a=d['a']; d_1=d[1]; d_2=d[2]

e = d["a"]()
f = d[1]()
