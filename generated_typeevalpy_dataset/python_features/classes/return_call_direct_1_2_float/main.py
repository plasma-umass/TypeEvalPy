# A method returns a different method and that method is directly called.


class MyClass:
    def func2(self):
        return 90.29

    def func1(self):
        return self.func2


a = MyClass()
b = a.func1()()
