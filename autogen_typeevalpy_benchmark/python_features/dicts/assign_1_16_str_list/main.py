# A dictionary key is assigned to a function.


def func1():
    return 'bpmzi'


def func2():
    return [99, 49, 66]


d = {"a": func1}

d["a"] = func2

e = d["a"]()
