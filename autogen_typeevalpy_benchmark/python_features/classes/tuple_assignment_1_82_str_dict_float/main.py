# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return 'lhirt'

    def func2(self):
        return {'dxkps': 65, 'jcrow': 63, 'vyvog': 55}

    def func3(self):
        return 14.37


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
