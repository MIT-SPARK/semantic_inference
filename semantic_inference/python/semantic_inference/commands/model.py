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
"""Subcommands for working with onnx models."""

import click
import onnx

TYPES = [
    "unknown",
    "float32",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "int32",
    "int64",
    "string",
    "bool",
    "float16",
    "float64",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
]


def _get_repr_str(node):
    tensor_type = node.type.tensor_type
    if not tensor_type.HasField("shape"):
        return "unknown"
    values = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            values.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            values.append(str(d.dim_param))
        else:
            values.append("?")

    repr_str = f"{node.name} ({TYPES[tensor_type.elem_type]}): "
    return repr_str + " x ".join(values)


def _show_model(model):
    input_initializer = set([node.name for node in model.graph.initializer])
    input_nodes = [x for x in model.graph.input if x.name not in input_initializer]

    model_str = "\nInputs\n======\n"
    for node in input_nodes:
        model_str += f"  - {_get_repr_str(node)}\n"

    model_str += "\nOutputs\n=======\n"
    for node in model.graph.output:
        model_str += f"  - {_get_repr_str(node)}"

    return model_str


@click.group(name="model")
def cli():
    """Subcommands for examining models."""
    pass


@cli.command()
@click.argument("model_paths", nargs=-1, type=click.Path(exists=True))
def show(model_paths):
    """Show model information for each model in MODEL_PATHS."""
    for model_path in model_paths:
        click.secho(f"Loading model: {model_path}", fg="green")
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            click.secho(f"The model is invalid: {e}", fg="red")
            continue

        click.secho(_show_model(model))
