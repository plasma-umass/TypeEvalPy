# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return 93.53

    def func2(self):
        return 62

    def func3(self):
        return {'itvyt': 63, 'tican': 77, 'coqxv': 63}


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
