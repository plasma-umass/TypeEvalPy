# Class Variable is assigned to a variable
class MyClass:
    class_var = 'acfgr'

    def __init__(self, instance_var):
        self.instance_var = instance_var


a = MyClass([73, 86, 27])
b = a.class_var
