# Class Variable is assigned to a variable
class MyClass:
    class_var = 38

    def __init__(self, instance_var):
        self.instance_var = instance_var


a = MyClass('mwhnp')
b = a.class_var
