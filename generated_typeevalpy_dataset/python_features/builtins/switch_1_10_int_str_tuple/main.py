#  A function is defined with switch statement in it.
def func(value):
    match value:
        case 67:
            return 'bjjws'
        case 'bjjws':
            return 67
        case _:
            return "default"


a = func(67)
b = func('bjjws')
c = func((20, 35, 24))
