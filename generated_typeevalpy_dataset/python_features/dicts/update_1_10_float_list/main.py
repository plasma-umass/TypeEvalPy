# The update method of dictionaries is used.


def func1():
    return [35, 25, 39]


def func2():
    return 59.97


d = {"a": func1}

d.update({"a": func2})
e = d["a"]()
