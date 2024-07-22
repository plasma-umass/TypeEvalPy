# A class is instantiated and we call one of its functions. The function called assigns self to a variable and using that variable we call a different function contained in the class
class MyClass:
    def func1(self):
        return (1, 88, 27)

    def func2(self):
        a = self
        return a.func1()


a = MyClass()
b = a.func2()
