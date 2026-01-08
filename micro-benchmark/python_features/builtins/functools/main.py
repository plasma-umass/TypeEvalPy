# A program demonstrating the use of the reduce() function from the functools module in Python.
from functools import reduce


def multiply(x, y):
    return x * y


numbers = [1, 2]; numbers_0=numbers[0]; numbers_1=numbers[1]
product = reduce(multiply, numbers)
