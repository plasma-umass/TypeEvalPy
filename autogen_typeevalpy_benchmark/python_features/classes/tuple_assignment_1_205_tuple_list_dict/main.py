# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return (9, 20, 91)

    def func2(self):
        return [68, 89, 99]

    def func3(self):
        return {'ljxhd': 76, 'qmapa': 72, 'juegr': 3}


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
