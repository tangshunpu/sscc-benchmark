from sscc.cli.benchmark import format_exception


def test_exception_text_is_bounded_for_csv():
    text = format_exception(RuntimeError("x" * 5000), max_chars=100)
    assert text.startswith("RuntimeError: ")
    assert "truncated" in text
    assert len(text) < 160


def test_short_exception_is_preserved():
    assert format_exception(ValueError("bad")) == "ValueError: bad"
