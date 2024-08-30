"""Test that configuration structs work as expected."""

import semantic_inference
from dataclasses import dataclass
from typing import Any


@semantic_inference.register_config("test", name="foo")
@dataclass
class Foo(semantic_inference.Config):
    """Test configuration struct."""

    a: float = 5.0
    b: int = 2
    c: str = "hello"


@semantic_inference.register_config("test", name="bar")
@dataclass
class Bar(semantic_inference.Config):
    """Test configuration struct."""

    bar: str = "world"
    d: int = 15


@dataclass
class Parent(semantic_inference.Config):
    """Test configuration struct."""

    child: Any = semantic_inference.config_field("test", default="foo")
    param: float = -1.0


def test_dump():
    """Make sure dumping works as expected."""
    foo = Foo()
    result = foo.dump()
    expected = {"a": 5.0, "b": 2, "c": "hello"}
    assert result == expected

    bar = Bar()
    result = bar.dump()
    expected = {"bar": "world", "d": 15}
    assert result == expected

    parent = Parent()
    result = parent.dump()
    expected = {"child": {"type": "foo", "a": 5.0, "b": 2, "c": "hello"}, "param": -1.0}
    assert result == expected


def test_update():
    """Test that update works as expected."""
    foo = Foo()
    assert foo == Foo()

    # empty update does nothing
    foo.update({})
    assert foo == Foo()

    foo.update({"a": 10.0})
    assert foo == Foo(a=10.0)

    foo.update({"b": 1, "c": "world"})
    assert foo == Foo(a=10.0, b=1, c="world")

    parent = Parent()
    parent.update({"child": {"b": 1, "c": "world"}, "param": -2.0})
    assert parent == Parent(child=Foo(a=5.0, b=1, c="world"), param=-2.0)


def test_save_load(tmp_path):
    """Test that saving and loading works."""
    filepath = tmp_path / "config.yaml"
    parent = Parent(child=Bar(bar="hello", d="2.0"), param=-2.0)
    parent.save(filepath)
    result = semantic_inference.Config.load(Parent, filepath)
    assert parent == result


def test_factory():
    """Test that the factory works."""
    registered = semantic_inference.ConfigFactory.registered()
    assert len(registered) > 0
    assert "test" in registered
    registered_names = [x[0] for x in registered["test"]]
    assert "foo" in registered_names
    assert "bar" in registered_names
