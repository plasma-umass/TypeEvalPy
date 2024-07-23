# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return {'stixu': 97, 'usupj': 80, 'iuxye': 97}


class B:
    def func(self):
        return 80


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return 25.19


c = C()
d = c.func()
