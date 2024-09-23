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
