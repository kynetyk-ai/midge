from pi import __version__


def test_version_present() -> None:
    assert isinstance(__version__, str)
