# Use a nested dictionary and assign a value to it.


def func1():
    return 42


def func2():
    return "Hello from func2"


d = {"a": {"b": func1}}; d_a=d['a']; d_a_b=d['a']['b']

d["a"]["b"] = func2; d_a_b__2=d['a']['b']

e = d["a"]["b"]()
func1()
