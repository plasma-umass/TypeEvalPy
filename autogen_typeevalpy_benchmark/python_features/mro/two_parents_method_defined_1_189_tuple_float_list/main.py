# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return (95, 96, 77)


class B:
    def func(self):
        return 59.05


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return [43, 63, 24]


c = C()
d = c.func()
