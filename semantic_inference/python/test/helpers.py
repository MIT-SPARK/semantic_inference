"""Various test helpers that don't work as fixtures."""

import torch
import pytest
import functools
import importlib


GPU_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU_SKIP)]
parametrize_device = pytest.mark.parametrize("device", DEVICES)


def _test_module(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def validate_modules(*args):
    """Check that all modules exist and are importable."""
    return functools.reduce(lambda x, y: x and y, [_test_module(x) for x in args], True)
