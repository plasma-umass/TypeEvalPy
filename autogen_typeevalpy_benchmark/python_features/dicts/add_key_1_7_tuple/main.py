# A new key-value is added to the dictionary which is a function.


def func():
    return (77, 34, 44)


d = {}

d["b"] = func
e = d["b"]()
