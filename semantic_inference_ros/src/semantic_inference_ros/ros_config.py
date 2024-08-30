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
