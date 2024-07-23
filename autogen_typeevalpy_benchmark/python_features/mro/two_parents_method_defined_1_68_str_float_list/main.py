# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return 'faqxp'


class B:
    def func(self):
        return 79.02


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return [17, 12, 4]


c = C()
d = c.func()
