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
"""Module containing ros config parsing infrastructure."""
from semantic_inference import Config
import pathlib
import rospy


# TODO(nathan) test this to make sure list and dictionary behavior is correct
def load_ros_params(ns=""):
    """Load all params."""

    def _relative_to(ns, name):
        if ns == "":
            return name

        return str(pathlib.Path(name).relative_to(pathlib.Path(ns)))

    def _insert_nested(params, name, value):
        if len(name) and name[0] == "/":
            name = name[1:]

        if "/" not in name:
            if name not in params:
                params[name] = value
            elif isinstance(value, list):
                params[name] += value
            elif isinstance(value, dict):
                params[name].update(value)
            else:
                params[name] = value

            return

        parts = name.split("/")
        curr, rest = parts[0], parts[1:]
        if curr not in params:
            params[curr] = {}

        _insert_nested(params[curr], "/".join(rest), value)

    resolved = "" if ns == "" else rospy.resolve_name(ns)
    param_names = rospy.get_param_names()
    params = {}
    for name in param_names:
        if resolved == "" or name.find(resolved) == 0:
            _insert_nested(params, _relative_to(resolved, name), rospy.get_param(name))

    return params


def load_from_ros(cls, ns=""):
    """Populate a config from ros params."""
    assert issubclass(cls, Config), f"{cls} is not a config!"

    instance = cls()
    params = load_ros_params(ns)
    instance.update(params)
    return instance
