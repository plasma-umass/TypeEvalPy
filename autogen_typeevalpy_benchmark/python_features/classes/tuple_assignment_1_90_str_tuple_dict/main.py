# We perform tuple assignment on class methods.


class MyClass:
    def __init__(self):
        pass

    def func1(self):
        return 'spwxi'

    def func2(self):
        return (52, 84, 55)

    def func3(self):
        return {'xqfee': 88, 'kxsfc': 69, 'lagbf': 29}


class MyClass2:
    def __init__(self):
        pass


a, b = MyClass(), MyClass2()

c, (d, e) = a.func1, (a.func2, a.func3)

f = c()
g = d()
h = e()
