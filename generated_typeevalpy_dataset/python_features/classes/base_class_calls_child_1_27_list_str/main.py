# The base class calls a function defined by the child class.
class A:
    def func(self):
        return self.child()


class B(A):
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return [82, 40, 81]


class C(A):
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 'ymowi'


b = B()
d = b.func()

c = C()
e = c.func()
