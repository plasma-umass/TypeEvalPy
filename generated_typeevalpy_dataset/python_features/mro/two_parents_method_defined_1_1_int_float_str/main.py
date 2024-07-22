# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return 76


class B:
    def func(self):
        return 38.54


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return 'plich'


c = C()
d = c.func()
