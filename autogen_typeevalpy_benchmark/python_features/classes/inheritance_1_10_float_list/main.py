# Calling methods of inherited class
class MyClass:
    def func(self):
        return 12.9


class MySubClass(MyClass):
    def sub_func(self):
        return [21, 54, 78]


a = MySubClass()
b = a.func()
c = a.sub_func()
