# A simple function with a decorator.


def dec(f):
    def wrapper():
        return (11, 19, 76)

    return wrapper


@dec
def func():
    pass


a = func()
