# Class Variable is assigned to a variable
class MyClass:
    class_var = 84.57

    def __init__(self, instance_var):
        self.instance_var = instance_var


a = MyClass(4)
b = a.class_var
