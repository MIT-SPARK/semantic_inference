"""Quick script to read an onnx file and print information."""
import onnx  # onnx required downgrading numpy from 1.19.5 to 1.19.4
import sys

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


def main():
    """Load model and show information."""
    # load model
    if len(sys.argv) != 2:
        print("Error! Must provide path to .onnx file as argument.")
        sys.exit(1)

    print(f"Loading model: {sys.argv[1]}")
    model = onnx.load(sys.argv[1])

    # check if model valid
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
        sys.exit(1)

    print("The model is valid!")

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


if __name__ == "__main__":
    main()
