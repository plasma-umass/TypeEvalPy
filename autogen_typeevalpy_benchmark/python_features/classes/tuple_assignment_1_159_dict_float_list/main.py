# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return {'jhbjn': 88, 'fihbd': 8, 'tfhcc': 56}

    def func2(self):
        return 14.1

    def func3(self):
        return [83, 95, 46]


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
