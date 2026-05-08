"""A pocket calculator with one bug."""


def add(a, b):
    return a + b


def subtract(a, b):
    # BUG: arguments are swapped
    return b - a


def multiply(a, b):
    return a * b
