# Functions are assigned as elements of a list and then called.


def func1():
    return [93, 93, 95]


def func2():
    return 'sqwmg'


def func3():
    return 66.3


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'hhfhe': 4, 'hqxhh': 86, 'rlvby': 74}


b = ["Hello"]
b[0] = func4

f = b[0]()
