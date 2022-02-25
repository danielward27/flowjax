from jaxflows.train_utils import count_fruitless
import pytest

def test_count_fruitless():
    assert count_fruitless([12, 2, 3, 4]) == 2
    assert count_fruitless([0]) == 0
    assert count_fruitless([0, 12]) == 1