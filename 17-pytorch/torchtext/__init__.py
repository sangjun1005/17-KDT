"""Local torchtext compatibility package for PyTorch 2.7 notebooks."""

from torchtext_compat import __version__


def disable_torchtext_deprecation_warning():
    """Compatibility no-op matching torchtext 0.18's public API."""


__all__ = ["__version__", "disable_torchtext_deprecation_warning"]
