"""Tiny stats library with one bug."""


def mean(xs):
    """Arithmetic mean of a list of numbers."""
    if not xs:
        raise ValueError("empty list")
    # BUG: divides by len(xs) - 1 instead of len(xs)
    return sum(xs) / (len(xs) - 1)


def variance(xs):
    """Population variance."""
    if not xs:
        raise ValueError("empty list")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)
