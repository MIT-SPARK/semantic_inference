# Semantic Recolor Utilities

Installation requires Cuda and TensorRT:

To install a minimal setup (after adding the Cuda repositories):
```
sudo apt install cuda-libraries-11-1 cuda-libraries-dev-11-1 cuda-nvrtc-dev-11-1 \
                 libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev cuda-nvcc-11-1
```

You may have to add the cuda libraries and cuda binaries to your path, e.g. in `.zshrc`:
```
export PATH=$PATH:/usr/local/cuda-11-1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11-1/lib64
```

For *most* of the scripts:

```
python3 -m venv semantic_recolor  # or whatever you want
source semantic_recolor/bin/activate
cd /path/to/semantic_recolor
pip install --upgrade pip
pip install wheel
pip install -r scripts/requirements.txt
```

## Models

Several pre-exported models live [here](https://drive.google.com/drive/folders/1GrmgFDFCssDxKe_Nyx8PPTK1pRMA0gEO?usp=sharing)

You can export a model from the MIT scene parsing challenge via [this script](scripts/export_onnx_model.py).

You will need to install a couple more dependencies:
```
source semantic_recolor/bin/activate
pip install torch torchvision
pip install git+https://github.com/CSAILVision/semantic-segmentation-python.git@master
```

A good naming scheme for exported models is `{model_name}_{image_height}_{image_width}_{onnx_instruction_version}.yaml`.

To check if the model is valid and show input/output names, run [this](scripts/check_onnx_model.py) script.

When exporting a new model, you will also need to write a config similar to [this](config/hrnetv2_360_640_v12.yaml).

To use the new model, pass in the appropriate argument (`model_name`) to the launch file being used.

## New Datasets

To adapt to a new dataset (or new set of labels), you will have to:

  - Make a new grouping config (see [this](config/label_groupings/ade150_outdoor.yaml) or [this](config/label_groupings/ade150_indoor.yaml) for examples)
  - Run [this](scripts/make_color_config.py) to export the color configuration. A typical invocation is `python scripts/make_color_config.py config/label_groupings/new_config.yaml config/colors/` from the root repo directory with your environment sourced.
  - Pass in the appropriate arguments to the launch file (`dataset_name`)

You can view the groupings for a particular category label space by running [this](scripts/show_label_groupings.py).
A typical invocation is `python scripts/show_label_groupings.py resources/ade20k_categories.csv config/label_groupings/ade150_indoor.yaml` or `python scripts/show_label_groupings.py resources/mpcat40.tsv config/label_groupings/mpcat40_objects.yaml -n 1`.

You can also view the groupings for a particular color csv files via [this](scripts/show_csv_groupings.py).
For most color configs exported by this package, the group names will be one-to-one with the category labels.
