def test_import_package():
    import portfolio_fdc

    assert hasattr(portfolio_fdc, "__version__")


def test_import_subpackages():
    pass
