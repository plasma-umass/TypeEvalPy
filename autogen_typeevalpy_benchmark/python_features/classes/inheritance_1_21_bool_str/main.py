# Calling methods of inherited class
class MyClass:
    def func(self):
        return False


class MySubClass(MyClass):
    def sub_func(self):
        return 'lrbde'


a = MySubClass()
b = a.func()
c = a.sub_func()
