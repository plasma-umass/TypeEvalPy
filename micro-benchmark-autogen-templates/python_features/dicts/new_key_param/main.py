# A new key is added to a dictionary using the parameter of a function as a key.


def func2():
    return "Hello from func2"


def func(key="a"):
    d[key] = func2


d = {}

func()
e = d["a"]()
