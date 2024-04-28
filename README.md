# Semantic Recolor Utilities

Installation requires Cuda and TensorRT. You can add the Cuda repositories [here](https://developer.nvidia.com/cuda-downloads) by installing the `deb (network)` package or
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Install the latest nvinfer and minimal cuda dependencies (you should check what version of cuda gets installed with TensorRT):
```
sudo apt install libnvinfer-dev libnvonnxparsers-dev libnvinfer-plugin-dev cuda-nvcc-12-4
```

Intalling a previous version:
```
export TRT_VERSION=8.6.1.6-1+cuda12.0
sudo apt install libnvinfer-dev=$TRT_VERSION libnvonnxparsers-dev=$TRT_VERSION libnvinfer-plugins-dev=$TRT_VERSION libnvinfer-headers-dev=$TRT_VERSION libnvinfer-headers-plugin-dev=$TRT_VERSION cuda-nvcc-12-4
```

## Python scripts

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

## Pre-recorded Semantics

You can produce a rosbag of semantic images and labels using [this](scripts/make_rosbag.py). The script can be invoked like this:

```
python scripts/make_rosbag.py path/to/input/bag rgb_topic [--is-compressed]
```

The script reads every image in the input bag for the provided topic (which can optionall  be a compressed image) and then sends the image to a remote "inference" server using zmq and gets back a label image which it writes to the output bag.
See [this](third_party/one_former.py) for an example using oneformer.

By default, the produced bag has both the original labels and a color image using the provided color config.
You can use the semantically colored image directly like you would for the normal output of the online nodelet.
Alternaltively, you can use the recolor nodelet to recolor the labels online (especially if you want to change what labels map to what colors), see [this](launch/recolor_pointcloud.launch) for more details.
