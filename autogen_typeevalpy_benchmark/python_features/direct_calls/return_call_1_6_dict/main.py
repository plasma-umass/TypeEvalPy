# A function `func` is called and returns a function `return_func` which is later called directly in the form func()().


def return_func():
    def nested_return_func():
        return {'fqqxq': 76, 'chezj': 15, 'zlojr': 93}

    return nested_return_func


def func():
    return return_func


a = func()()
b = func()()()
