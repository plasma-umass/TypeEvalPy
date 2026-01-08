# Check if tool type coerces integer and string values.


def func1():
    return "Hello from func1"


def func2():
    return 42


d = {1: func1, "1": func2}; d_1=d[1]; d_1__2=d['1']

e = d[1]()
f = d["1"]()
