# A dictionary is returned.


def func2():
    return "Hello from func2"


def func1():
    d = {"a": func2}; d_a=d['a']
    return d


b = func1()
c = b["a"]()
