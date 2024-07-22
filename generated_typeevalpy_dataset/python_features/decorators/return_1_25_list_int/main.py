# The decorator is a function call.


def func1():
    def dec(f):
        return inner

    def inner():
        return [93, 72, 26]

    return dec


@func1()
def func2():
    return 78


a = func2()
