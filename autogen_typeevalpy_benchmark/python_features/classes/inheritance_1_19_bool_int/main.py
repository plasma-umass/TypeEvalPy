# Calling methods of inherited class
class MyClass:
    def func(self):
        return True


class MySubClass(MyClass):
    def sub_func(self):
        return 63


a = MySubClass()
b = a.func()
c = a.sub_func()
