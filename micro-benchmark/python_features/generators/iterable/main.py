# Test that all the methods of a generator are called.


class func:
    def __init__(self, n):
        self.n = n
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num + 1
            return cur
        else:
            raise StopIteration()


output_list = [i for i in func(3)]; output_list_0=output_list[0]; output_list_1=output_list[1]; output_list_2=output_list[2]
