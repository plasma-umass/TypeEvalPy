# A function defined inside another function's context serves as a decorator.


def func():
    def dec(f):
        return modified_inner

    def modified_inner():
        return 10.9

    @dec
    def inner():
        return [9, 86, 65]

    return inner()


a = func()
