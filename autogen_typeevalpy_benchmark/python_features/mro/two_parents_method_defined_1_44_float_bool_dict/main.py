# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return 10.7


class B:
    def func(self):
        return False


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return {'ylqjs': 74, 'zwyzt': 80, 'vcgao': 99}


c = C()
d = c.func()
