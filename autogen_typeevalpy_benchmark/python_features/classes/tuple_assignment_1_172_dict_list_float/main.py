# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return {'vwnjt': 54, 'loajq': 63, 'tfxpr': 6}

    def func2(self):
        return [85, 39, 36]

    def func3(self):
        return 78.72


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
