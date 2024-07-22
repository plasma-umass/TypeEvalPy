# Class with a decorator.


def my_decorator(cls):
    class NewClass(cls):
        def my_method(self):
            return 10

    return NewClass


@my_decorator
class MyClass:
    def my_method(self):
        return [20, 44, 88]


a = MyClass()
b = a.my_method()
