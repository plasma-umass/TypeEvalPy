# A simple function with a decorator.


def dec(f):
    def wrapper():
        return 17

    return wrapper


@dec
def func():
    pass


a = func()
