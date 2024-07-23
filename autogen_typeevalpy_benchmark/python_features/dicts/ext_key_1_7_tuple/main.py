# A key value is imported from an external module and used to access a function stored in a dictionary.


from ext import key


def func():
    return (13, 7, 24)


d = {"a": func}

e = d[key]()
