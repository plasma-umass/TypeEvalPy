# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return 31.36

    def func2(self):
        return {'pprhu': 89, 'xwsoh': 23, 'sppcw': 90}

    def func3(self):
        return (78, 24, 39)


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
