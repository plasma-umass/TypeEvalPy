# Class Variable is assigned to a variable
class MyClass:
    class_var = 'ljqdr'

    def __init__(self, instance_var):
        self.instance_var = instance_var


a = MyClass(3.3)
b = a.class_var
