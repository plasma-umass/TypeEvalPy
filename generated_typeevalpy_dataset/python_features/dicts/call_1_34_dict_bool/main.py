# A dictionary containing functions as values is created.


def func1():
    return {'lqcba': 14, 'bogwq': 12, 'hfisk': 72}


def func2():
    return True


d = {"a": func1, 1: func2, 2: 3}

e = d["a"]()
f = d[1]()
