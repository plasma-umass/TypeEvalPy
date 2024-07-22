# Define function with defaults being other functions.
# Then call these functions, passing different functions as keyword arguments.


def func3():
    return {'gnnsi': 56, 'ajlln': 49, 'vujng': 83}


def func2(a=func3):
    return a()


def func1(a, b=func2):
    return a(b)


c = func1(a=func2, b=func3)
