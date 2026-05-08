from stats import mean, variance


def test_mean_basic():
    assert mean([1, 2, 3, 4, 5]) == 3.0


def test_mean_two_elements():
    assert mean([2, 4]) == 3.0


def test_variance_zero():
    assert variance([5, 5, 5]) == 0
