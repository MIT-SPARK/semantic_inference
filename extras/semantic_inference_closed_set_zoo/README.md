# Closed Set Model Zoo

This package contains wrappers around closed-set segmentation models that can't easily be exported to ONNX.
At the moment this is just Mask2Former, but it may grow to contain others (OneFormer, etc.)

> :warning: **Warning** </br>
> This package is somewhat unstable as many of the transformer-based segmentation models have custom operators and don't actually package the model or code in a way that is useable by others.

## Installation

This assumes you've installed `semantic_inference` into a virtual environment already.
The following (assuming you're at the top level of the `semantic_inference` repository and you've sourced the virtual environment) should work:
```shell
pip install ninja  # makes detectron2 build and installation faster
cd extras/semantic_inference_closed_set_zoo
pip install .
```

> **Note** </br>
> You way want to use the `--no-build-isolation` flag to get around having pip install pytorch into the temporary environment that builds this package, which works but takes a long time.
> You may need to install various CUDA libraries that match the CUDA version you have installed (e.g., `sudo apt install cuda-libraries-dev-12-5` was required on my machine though you might be able to get away with less).
> This looks like `pip install --no-build-isolation ./extras/semantic_inference_closed_set_zoo`
