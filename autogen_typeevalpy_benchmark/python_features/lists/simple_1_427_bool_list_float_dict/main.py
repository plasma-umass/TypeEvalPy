# Functions are assigned as elements of a list and then called.


def func1():
    return False


def func2():
    return [86, 68, 22]


def func3():
    return 38.85


a = [func1, func2, func3]

c = a[0]()
d = a[1]()
e = a[2]()


def func4():
    return {'xxtch': 57, 'ftrbt': 35, 'obikc': 61}


b = ["Hello"]
b[0] = func4

f = b[0]()
