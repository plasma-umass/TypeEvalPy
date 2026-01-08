# A dictionary key is assigned to a function.


def func1():
    return "Hello from func1"


def func2():
    return 42


d = {"a": func1}; d_a=d['a']

d["a"] = func2; d_a__2=d['a']

e = d["a"]()
func1()
