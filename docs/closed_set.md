# Closed-Set Segmentation

## Setting up

### Getting Dependencies

Using dense 2D (closed-set) semantic-segmentation models requires CUDA and TensorRT.
The process for installing CUDA and TensorRT varies by system.
Most users should follow NVIDIA's documentation [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

> **Note**<br>
> The packages in this repository will compile without TensorRT and CUDA, and some functionality (e.g., remapping label images and visualizing the results) is available without TensorRT and CUDA.

In some cases, a more minimal installation is desirable (e.g., containers).  The following steps *should* ensure a minimum viable installation of TensorRT:

  1. Add the CUDA repositories [here](https://developer.nvidia.com/cuda-downloads) by installing the `deb (network)` package or

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

  2. Check what version of CUDA TensorRT is built against:

```console
$ apt search nvinfer | grep cuda
libnvinfer-bin/unknown 10.4.0.26-1+cuda12.6 amd64
libnvinfer-dev/unknown,now 10.4.0.26-1+cuda12.6 amd64
...
```

  3.  Install TensorRT and CUDA if necessary:

```bash
# use the corresponding version number from the previous step or omit nvcc if already installed
sudo apt install libnvinfer-dev libnvonnxparsers-dev libnvinfer-plugin-dev cuda-nvcc-12-6
```

### Building

Once the necessary dependencies are installed and this repository has been placed in a workspace, run the following:
```
catkin build
```

You can run the following to make sure everything is working:
```
catkin test semantic_inference
```

### Models

Running dense 2D semantic segmentation requires obtaining a pre-trained model.
Several pre-exported models live [here](https://drive.google.com/drive/folders/1GrmgFDFCssDxKe_Nyx8PPTK1pRMA0gEO?usp=sharing).

> **Note** <br>
> We recommend using models within the [dense2d](https://drive.google.com/drive/folders/17p_ZZIxI9jI_3GjjtbMijC2WFnc9Bz-a?usp=sharing) folder, which are named corresponding to the labelspace they output to.
> The top-level models are deprecated as they do not follow this naming scheme (they all output to the ade20k label space).

All models should be placed in the `models` directory of the `semantic_inference` package in local clone of the repository.
To use a specific downloaded model, use the argument `model_name:=MODEL_NAME` when running the appropriate launch file (where `MODEL_NAME` is the filename of the model to use minus the extension).

Note that the pipeline as implemented works with any pre-trained model exported to onnx as long as the model takes 32-bit float 3-channel tensors as input and outputs labels in a single-channel tensor as integers.
The pipeline can optionally rescale and offset the input tensor.
See [here](exporting.md) for details on exporting a new model.

---

# Outdated Instructions

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

The script reads every image in the input bag for the provided topic (which can optionally be a compressed image) and then sends the image to a remote "inference" server using zmq and gets back a label image which it writes to the output bag.
See [this](third_party/one_former.py) for an example using oneformer.

By default, the produced bag has both the original labels and a color image using the provided color configuration.
You can use the semantically colored image directly like you would for the normal output of the online nodelet.
Alternatively, you can use the recolor nodelet to recolor the labels online (especially if you want to change what labels map to what colors), see [this](launch/recolor_pointcloud.launch) for more details.
