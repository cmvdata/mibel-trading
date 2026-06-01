"""Smoke tests verifying package imports cleanly."""

import mibel_trading


def test_package_version() -> None:
    """Package exposes a version string."""
    assert isinstance(mibel_trading.__version__, str)
    assert len(mibel_trading.__version__) > 0


def test_subpackages_importable() -> None:
    """All declared subpackages import without errors."""
    from mibel_trading import agents, benchmarks, data, env, eval, strategies

    assert all(
        mod is not None
        for mod in [agents, benchmarks, data, env, eval, strategies]
    )
