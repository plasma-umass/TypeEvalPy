# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return {'cbznd': 47, 'kubde': 20, 'deych': 44}

    def func2(self):
        return 'jtrij'

    def func3(self):
        return 24.07


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
