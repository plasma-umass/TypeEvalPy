# Class with a decorator.


def my_decorator(cls):
    class NewClass(cls):
        def my_method(self):
            return (25, 52, 59)

    return NewClass


@my_decorator
class MyClass:
    def my_method(self):
        return [68, 88, 23]


a = MyClass()
b = a.my_method()
