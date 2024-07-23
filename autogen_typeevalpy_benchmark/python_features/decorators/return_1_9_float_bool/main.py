# The decorator is a function call.


def func1():
    def dec(f):
        return inner

    def inner():
        return 37.91

    return dec


@func1()
def func2():
    return False


a = func2()
