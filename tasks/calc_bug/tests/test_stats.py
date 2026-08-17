import pytest

from calc.stats import clamp, mean, median, variance


def test_mean_of_integers():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_of_single_value():
    assert mean([7]) == 7


def test_mean_rejects_empty():
    with pytest.raises(ValueError):
        mean([])


def test_median_odd_length():
    assert median([3, 1, 2]) == 2


def test_median_even_length():
    assert median([4, 1, 3, 2]) == 2.5


def test_median_rejects_empty():
    with pytest.raises(ValueError):
        median([])


def test_variance_of_uniform_values():
    assert variance([2, 2, 2]) == 0


def test_variance_known_value():
    assert variance([1, 2, 3, 4]) == pytest.approx(1.25)


def test_clamp_inside_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_and_above():
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10


def test_clamp_rejects_inverted_range():
    with pytest.raises(ValueError):
        clamp(1, 10, 0)
