# Closed-Set Segmentation

## Setting up

### Getting Dependencies

Using dense 2D (closed-set) semantic-segmentation models requires CUDA and TensorRT.
The process for installing CUDA and TensorRT varies by system.
Most users should follow NVIDIA's documentation [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

> **Note**<br>
> The packages in this repository will compile without TensorRT and CUDA, and some functionality (e.g., remapping label images and visualizing the results) is available without TensorRT and CUDA.
> If you do not have a NVIDIA GPU, you can skip straight to [building](#building).

In some cases, a more minimal installation is desirable (e.g., containers).  The following steps *should* ensure a minimum viable installation of TensorRT:

  1. Add the CUDA repositories [here](https://developer.nvidia.com/cuda-downloads) by installing the `deb (network)` package or

```shell
# make sure you pick the correct ubuntu version!
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
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

```shell
# use the corresponding version number from the previous step or omit nvcc if already installed
sudo apt install libnvinfer-dev libnvonnxparsers-dev libnvinfer-plugin-dev cuda-nvcc-12-6
```

### Building

Once the necessary dependencies are installed and this repository has been placed in a workspace, run the following:
```shell
colcon build
```

You can run the following to validate that `semantic_inference` built correctly:
```shell
colcon test --packages-select semantic_inference
```

### Models

Running dense 2D semantic segmentation requires obtaining a pre-trained model.
Several pre-exported models live [here](https://drive.google.com/drive/folders/1GrmgFDFCssDxKe_Nyx8PPTK1pRMA0gEO?usp=sharing).
By default, the code uses [this](https://drive.google.com/file/d/1XRcsyLSvqqhqNIaOI_vmqpUpmBT6gk9-/view?usp=drive_link) model.

> **Note** <br>
> We recommend using models within the [dense2d](https://drive.google.com/drive/folders/17p_ZZIxI9jI_3GjjtbMijC2WFnc9Bz-a?usp=sharing) folder, which are named corresponding to the labelspace they output to.
> The top-level models are deprecated as they do not follow this naming scheme (they all output to the ade20k label space).

By default, the closed set node looks under the directory `$HOME/.semantic_inference` for models (this works on Linux or as long as `HOME` is set).
It is possible to change this directory by specifying the `SEMANTIC_INFERENCE_MODEL_DIR` environment variable.
To use a specific downloaded model, use the argument `model_file:=MODEL_FILE` when running the appropriate launch file (where `MODEL_NAME` is the filename of the model relative to the configured model directory).
Specifying an absolute filepath will override the default model directory.

Note that the pipeline as implemented works with any pre-trained model exported to [onnx](https://onnx.ai/) as long as the model takes 32-bit float 3-channel tensors as input and outputs labels in a single-channel tensor as integers.
The pipeline can optionally rescale and offset the input tensor.
See [here](exporting.md) for details on exporting a new model.

### Python utilities

You may find it useful to set up some of the included model utilities. From the top-level of this repository, run:
```shell
python3 -m virtualenv <DESIRED_PATH_TO_ENVIRONMENT>
source <DESIRED_PATH_TO_ENVIRONMENT>/bin/activate
pip install ./semantic_inference
```

## Using closed-set segmentation online

To use the open-set segmentation as part of a larger system, include [closed_set.launch.yaml](../semantic_inference_ros/launch/closed_set.launch.yaml) in your launch file. Often this will look like this:
```yaml
launch:
  # ... rest of launch file ...
  - set_remap: {from: "color/image_raw", to: "YOUR_INPUT_TOPIC_HERE"}
  - include: {file: "$(find-pkg-share semantic_inference_ros)/launch/closed_set.launch.yaml"}
```

> **Note** </br>
> You'll probably also want to namespace the included launch file and corresponding remap via a `group` tag and `push_ros_namespace` with the camera name as the namespace.

## Adding New Datasets

To adapt to a new dataset (or to make a new grouping of labels), you will have to:

  - Make a new grouping config (see [this](../semantic_inference_ros/config/label_groupings/ade20k_outdoor.yaml) or [this](../semantic_inference_ros/config/label_groupings/ade20k_indoor.yaml) for examples)
  - Pass in the appropriate arguments to the launch file (`labelspace_name`)

You can view the groupings for a particular labelspace by running `semantic-inference labelspace compare`.
For a grouping of the ade20k labelspace:
```shell
source <DESIRED_PATH_TO_ENVIRONMENT>/bin/activate
cd <PATH_TO_REPO>
semantic-inference labelspace compare semantic_inference/resources/ade20k_categories.csv semantic_inference_ros/config/label_groupings/ade20k_indoor.yaml
```
