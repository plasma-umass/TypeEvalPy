# Functions are assigned as elements of a list and then called.


def func1():
    return 'uocus'


def func2():
    return [56, 70, 89]


def func3():
    return (24, 22, 33)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'ntihf': 95, 'khoxp': 61, 'uhcpx': 5}


b = ["Hello"]
b[0] = func4

f = b[0]()
