import collective_bench


def test_package_import() -> None:
    assert collective_bench.__version__ == "0.1.0"
