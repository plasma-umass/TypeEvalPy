# A static function is defined and called.


class MyClass:
    @staticmethod
    def my_static_method(x, y):
        return x + y


result = MyClass.my_static_method(32.77, 32.77)
