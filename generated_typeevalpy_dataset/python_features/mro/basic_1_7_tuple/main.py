# A simple inheritance scheme.


class A:
    def func(self):
        return (1, 8, 57)


class B(A):
    pass


b = B()
c = b.func()
