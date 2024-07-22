# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return (93, 46, 68)


class B:
    def func(self):
        return 'aihea'


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return 88


c = C()
d = c.func()
