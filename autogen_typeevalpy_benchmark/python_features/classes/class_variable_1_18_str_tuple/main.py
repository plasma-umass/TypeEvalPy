# Class Variable is assigned to a variable
class MyClass:
    class_var = 'mpzuv'

    def __init__(self, instance_var):
        self.instance_var = instance_var


a = MyClass((72, 71, 78))
b = a.class_var
