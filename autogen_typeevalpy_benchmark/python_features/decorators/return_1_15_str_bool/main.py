# The decorator is a function call.


def func1():
    def dec(f):
        return inner

    def inner():
        return 'iltzh'

    return dec


@func1()
def func2():
    return True


a = func2()
