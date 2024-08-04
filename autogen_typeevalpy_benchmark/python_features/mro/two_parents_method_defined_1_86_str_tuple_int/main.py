# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return 'mdrid'


class B:
    def func(self):
        return (82, 41, 4)


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return 84


c = C()
d = c.func()
