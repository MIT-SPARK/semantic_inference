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
"""Base config class for config structures."""

import copy
import dataclasses
import pathlib
import pprint

import yaml

from semantic_inference.misc import Logger


class Config:
    """Base class for any config."""

    def dump(self, add_type=False):
        """Dump a config to a yaml-compatible dictionary."""
        values = dataclasses.asdict(self)
        info = ConfigFactory.get_info(self)
        if add_type and info:
            values["type"] = info[1]

        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if field.metadata.get("virtual_config", False):
                need_type = not isinstance(field_value, VirtualConfig)
                values[field.name] = field_value.dump(add_type=need_type)
            elif isinstance(field_value, Config):
                # recursion required because assdict won't dispatch dump
                values[field.name] = field_value.dump()

        return values

    def save(self, filepath):
        """Dump a class to disk."""
        filepath = pathlib.Path(filepath)
        with filepath.open("w") as fout:
            fout.write(yaml.safe_dump(self.dump()))

    def update(self, config):
        """Load settings from a dumped config."""
        if not isinstance(config, dict):
            Logger.error(f"Invalid config data provided to {self}")
            return

        for field in dataclasses.fields(self):
            prev = getattr(self, field.name)
            if isinstance(prev, Config) or field.metadata.get("virtual_config", False):
                prev.update(config.get(field.name, {}))
            else:
                setattr(self, field.name, config.get(field.name, prev))

    def show(self):
        """Show config in human readable format."""
        return f"{self.__class__.__name__}:\n{pprint.pformat(self.dump())}"

    @staticmethod
    def load(cls, filepath):
        """Load an abitrary config from file."""
        assert issubclass(cls, Config), f"{cls} is not a config!"

        instance = cls()
        with pathlib.Path(filepath).open("r") as fin:
            instance.update(yaml.safe_load(fin))

        return instance


class ConfigFactory:
    """Factory for configs."""

    __shared_state = None

    def __init__(self):
        """Make a factory."""
        # Borg pattern: set class state to global state
        if not ConfigFactory.__shared_state:
            ConfigFactory.__shared_state = self.__dict__
            self._factories = {}
            self._lookup = {}
            self._constructors = {}
        else:
            self.__dict__ = ConfigFactory.__shared_state

    @staticmethod
    def create(category, name, *args, **kwargs):
        """Instantiate a specific config."""
        instance = ConfigFactory()
        if category not in instance._factories:
            raise ValueError(f"No configs registered under category '{category}'")

        category_factories = instance._factories[category]
        if name not in category_factories:
            pass

        return category_factories[name](*args, **kwargs)

    @staticmethod
    def register(config_type, category, name=None, constructor=None):
        """Register a config type with the factory."""
        instance = ConfigFactory()
        if category not in instance._factories:
            instance._factories[category] = {}

        name = name if name else config_type.__name__
        instance._factories[category][name] = config_type
        instance._lookup[str(config_type)] = (category, name)
        if constructor:
            if category not in instance._constructors:
                instance._constructors[category] = {}
            instance._constructors[category][str(config_type)] = constructor

    @staticmethod
    def registered():
        """Get all registered types."""
        factories = ConfigFactory()._factories
        return {c: [(n, t) for n, t in fact.items()] for c, fact in factories.items()}

    @staticmethod
    def get_info(cls_instance):
        """Lookup category and name for class type."""
        typename = str(type(cls_instance))
        instance = ConfigFactory()
        return instance._lookup.get(typename)

    @staticmethod
    def get_constructor(category, name):
        """Get a constructor for a type if it exists."""
        instance = ConfigFactory()
        if category not in instance._factories:
            return None

        category_factories = instance._factories[category]
        if name not in category_factories:
            return None

        typename = str(category_factories[name])
        return instance._constructors[category][typename]


class VirtualConfig:
    """Holder for config type."""

    def __init__(self, category, default=None, required=True):
        """Make a virtual config."""
        self.category = category
        self.default = default
        self.required = required
        self._type = None
        self._config = None

    def dump(self, **kwargs):
        """Dump underlying config."""
        if not self._config:
            self._create(validate=False)
            if not self._config:
                return {}

        # TODO(nathan) this is janky
        values = self._config.dump()
        values["type"] = self._type
        return values

    def update(self, config_data):
        """Update config struct."""
        try:
            typename = config_data.get("type")
        except Exception:
            Logger.error(f"Could not get type for {self} from '{config_data}'")
            typename = None

        type_changed = typename is not None and self._type != typename
        if not self._config or type_changed:
            self._create(typename=typename)

        self._config.update(config_data)

    def create(self, *args, **kwargs):
        """Call constructor with config."""
        name = self._get_curr_typename()
        constructor = ConfigFactory.get_constructor(self.category, name)
        if not constructor:
            raise ValueError(f"constructor not specified for {self}")

        if not self._config:
            self._create()

        return constructor(self._config, *args, **kwargs)

    def _config_str(self):
        return f"<VirtualConfig(category={self.category}, type='{self._type}')>"

    def _get_curr_typename(self):
        return self._type if self._type else self.default

    def _create(self, typename=None, validate=True):
        name = typename if typename else self._get_curr_typename()
        if name is None and (self.required and validate):
            raise ValueError(f"Could not make virtual config for '{self.category}'")

        self._config = ConfigFactory.create(self.category, name)
        self._type = name

    def __repr__(self):
        """Access underlying repr."""
        return self._config_str() if not self._config else self._config.__repr__()

    def __eq__(self, other):
        """Access underlying eq."""
        return self._config == other if self._config else False

    def __deepcopy__(self, memo):
        """Copy virtual config."""
        new_config = VirtualConfig(
            self.category, default=self.default, required=self.required
        )
        if self._config:
            new_config._config = copy.deepcopy(self._config, memo)
            new_config._type = self._type

        return new_config

    def __getattr__(self, name):
        """Get underlying config field."""
        # TODO(nathan) clean this up
        assert name not in ["default", "category", "_type", "_config"]

        if not self._config:
            # create config if it doesn't exist
            self._config = self._create()

        return getattr(self._config, name)


def register_config(category, name="", constructor=None):
    """Register a class with the factory."""

    def decorator(cls):
        ConfigFactory.register(cls, category, name=name, constructor=constructor)
        return cls

    return decorator


def config_field(category, default=None, required=True):
    """Return a dataclass field."""

    def factory():
        return VirtualConfig(category, default=default, required=required)

    return dataclasses.field(default_factory=factory, metadata={"virtual_config": True})
