# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Fixtures for unit tests."""

import pytest
import pathlib
import imageio.v3 as iio


@pytest.fixture()
def resource_dir():
    """Get a path to the resource directory for tests."""
    return pathlib.Path(__file__).absolute().parent / "resources"


@pytest.fixture()
def suppress_torch():
    """Disable torch scientific notation."""
    import torch

    torch.set_printoptions(sci_mode=False, precision=3)
    yield
    torch.set_printoptions("default")


@pytest.fixture()
def cleanup_torch():
    """Cleanup torch memory."""
    import torch

    yield
    torch.cuda.empty_cache()


@pytest.fixture()
def image_factory():
    """Get images from resource directory."""
    import torch

    resource_path = pathlib.Path(__file__).absolute().parent / "resources"

    def factory(name):
        """Get image."""
        img_path = resource_path / name
        img = iio.imread(img_path)
        assert img is not None
        # convert to RGB order and torch
        return torch.from_numpy(img[:, :, ::-1].copy())

    return factory
