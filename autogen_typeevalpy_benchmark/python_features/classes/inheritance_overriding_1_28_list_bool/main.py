# Method Overriding by imherited class
class MyClass:
    def func(self):
        return [60, 71, 86]


class MySubClass(MyClass):
    def func(self):
        return True


a = MySubClass()
b = a.func()
