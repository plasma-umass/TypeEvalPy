# A simple function with a decorator.


def dec(f):
    def wrapper():
        return (100, 22, 44)

    return wrapper


@dec
def func():
    pass


a = func()
