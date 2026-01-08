# A generator is passed as a parameter.


def func(c):
    output_list = [i for i in c]; output_list_0=output_list[0]; output_list_1=output_list[1]; output_list_2=output_list[2]
    return output_list


class Cls:
    def __init__(self, max=0):
        self.max = max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.max:
            raise StopIteration

        result = 2**self.n
        self.n += 1
        return result


a = func(Cls(2)); a_0=a[0]; a_1=a[1]; a_2=a[2]
