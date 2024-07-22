# Functions are assigned as elements of a list and then called.


def func1():
    return [32, 9, 91]


def func2():
    return 10


def func3():
    return (23, 21, 71)


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'inkej': 32, 'ggafh': 53, 'hlxzx': 42}


b = ["Hello"]
b[0] = func4

f = b[0]()
