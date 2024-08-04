# A function defined inside another function's context serves as a decorator.


def func():
    def dec(f):
        return modified_inner

    def modified_inner():
        return True

    @dec
    def inner():
        return 32.29

    return inner()


a = func()
