# The update method of dictionaries is used.


def func1():
    return {'xqvnq': 63, 'yigox': 90, 'aclbx': 14}


def func2():
    return 88


d = {"a": func1}

d.update({"a": func2})
e = d["a"]()
