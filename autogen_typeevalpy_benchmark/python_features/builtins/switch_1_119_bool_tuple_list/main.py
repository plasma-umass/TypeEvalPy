#  A function is defined with switch statement in it.
def func(value):
    match value:
        case False:
            return (78, 31, 57)
        case (78, 31, 57):
            return False
        case _:
            return "default"


a = func(False)
b = func((78, 31, 57))
c = func([83, 60, 24])
