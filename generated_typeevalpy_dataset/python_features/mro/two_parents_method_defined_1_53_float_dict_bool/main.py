# A class is defined inheriting from two parents. However all the functions called are defined in the class itself.


class A:
    def __init__(self):
        pass

    def func(self):
        return 45.53


class B:
    def func(self):
        return {'yahvt': 89, 'ylyuk': 92, 'lbfpr': 78}


class C(A, B):
    def __init__(self):
        pass

    def func(self):
        return True


c = C()
d = c.func()
