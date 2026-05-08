from text_utils import reverse, uppercase_words


def test_reverse():
    assert reverse("abc") == "cba"


def test_uppercase_words_basic():
    assert uppercase_words("hello world") == "Hello World"


def test_uppercase_words_mixed_case():
    assert uppercase_words("hELLO wORLD") == "Hello World"


def test_uppercase_words_single():
    assert uppercase_words("ada") == "Ada"
