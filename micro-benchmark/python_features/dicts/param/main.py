# A dictionary is passed as a parameter.


def func2():
    return "Hello from func2"


def func1(d):
    return d["a"]()


d = {"a": func2}; d_a=d['a']

e = func1(d)
