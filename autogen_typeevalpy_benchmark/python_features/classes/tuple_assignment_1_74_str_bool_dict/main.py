# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return 'asaih'

    def func2(self):
        return False

    def func3(self):
        return {'ubfkm': 78, 'jwrue': 73, 'idcdm': 62}


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
