from network import sleep_and_redo


def test_succeeds_first_try():
    calls = {"n": 0}

    def op():
        calls["n"] += 1
        return "ok"

    assert sleep_and_redo(op) == "ok"
    assert calls["n"] == 1
