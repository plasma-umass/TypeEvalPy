# A class is instantiated and we assign one of its functions to a variable and then call that variable.
class MyClass:
    def func(self):
        return "Hello from func in MyClass"


a = MyClass()
b = a.func
b()
