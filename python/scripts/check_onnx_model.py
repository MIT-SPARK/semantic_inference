"""Quick script to read an onnx file and print information."""

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


def _show_model(model_path):
    click.secho(f"Loading model: {model_path}", fg="green")
    model = onnx.load(model_path)

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        click.secho(f"The model is invalid: {e}", fg="red")
        return

    # show input names
    input_initializer = set([node.name for node in model.graph.initializer])
    input_nodes = [
        node for node in model.graph.input if node.name not in input_initializer
    ]
    print("")
    print("Inputs")
    print("======")
    for node in input_nodes:
        print(f"  - {_get_repr_str(node)}")

    print("")
    print("Outputs")
    print("=======")
    for node in model.graph.output:
        print(f"  - {_get_repr_str(node)}")


@click.command()
@click.argument("model_paths", nargs=-1, type=click.Path(exists=True))
def main(model_paths):
    """Load model and show information."""
    for model_path in model_paths:
        _show_model(model_path)


if __name__ == "__main__":
    main()
