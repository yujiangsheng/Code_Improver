from helper import make
from core import NewName


def test_helper_returns_new_name_instance():
    assert isinstance(make(), NewName)


def test_hello():
    assert NewName().hello().startswith("hi from")
