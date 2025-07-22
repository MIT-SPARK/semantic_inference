# semantic_inference

<div align="center">
   <img src="docs/media/demo_segmentation.png"/>
</div>

This repository provides code for running inference on images with pre-trained models to provide both closed and open-set semantics.
Closed-set and open-set segmentation are implemented as follows:
  - Inference using dense 2D closed-set semantic segmentation models is implemented in c++ using TensorRT
  - Inference using open-set segmentation models and language features is implemented in python

Both kinds of semantic segmentation have a ROS interface associated with them, split between c++ and python as appropriate.
> **Note** </br>
> We have archived our ROS1 interface [here](https://github.com/MIT-SPARK/semantic_inference/tree/archive/ros_noetic) and do not plan on any additional ROS1 development.

## Table of Contents

- [Credits](#credits)
- [Filing Issues](#filing-issues)
- [Getting started](#getting-started)
  - [Closed-set](docs/closed_set.md#setting-up)
  - [Open-set](docs/open_set.md#setting-up)
- [Usage](#usage)

## Credits

`semantic_inference` was primarily developed by [Nathan Hughes](https://mit.edu/sparklab/people.html) at the [MIT-SPARK Lab](https://mit.edu/sparklab), assisted by [Yun Chang](https://mit.edu/sparklab/people.html), [Jared Strader](https://mit.edu/sparklab/people.html), [Aaron Ray](https://mit.edu/sparklab/people.html), and [Dominic Maggio](https://mit.edu/sparklab/people.html).
A full list of contributors is maintaned [here](contributors.md).
We welcome additional contributions!

## Filing Issues

Please understand that this is research code maintained by busy graduate students, **which comes with some caveats**:
  1. We do our best to maintain and keep the code up-to-date, but things may break or change occasionally
  2. We do not have bandwidth to help adapt the code to new applications
  3. The documentation, code-base and installation instructions are geared towards practitioners familiar with ROS

> **:warning: Warning**<br>
> We don't support other platforms. Issues requesting support on other platforms (e.g., Ubuntu 18.04, Windows) will be summarily closed.

Thank you in advance for your understanding!

## Getting started

The recommended use-case for the repository is with ROS.
We assume some familiarity with ROS in these instructions.
To start, clone this repository into your colcon workspace and run rosdep to get any missing dependencies.
This usually looks like the following:
```bash
cd /path/to/colcon_ws/src
git clone git@github.com:MIT-SPARK/semantic_inference.git
vcs import . < semantic_inference/install/packages.yaml
rosdep install --from-paths . --ignore-src -r -y
```

An (optional) quick primer for setting up a minimal workspace is below for those less familiar with ROS.

<details>

<summary>Making a workspace</summary>

First, make sure rosdep is setup:
```bash
# Initialize necessary tools for working with ROS
sudo apt install python3-vcstool python3-rosdep
sudo rosdep init
rosdep update
```

Then, make the workspace and initialize it:
```bash
# Setup the workspace
mkdir -p path/to/colcon_ws/src
cd colcon_ws
echo "build: {cmake-args: [--no-warn-unused-cli, -DCMAKE_BUILD_TYPE=Release]}" > colcon_defaults.yaml
```

</details>

Once you've added this repository to your workspace, follow one (or both) of the following setup-guides as necessary:
- [Closed-Set](docs/closed_set.md#setting-up)
- [Open-Set](docs/open_set.md#setting-up)

> **Note** </br>
> Some of our other (larger) packages have or will have more accessible guides to getting `semantic_inference` set up for specific applications, such as [Hydra](https://github.com/MIT-SPARK/Hydra), [Khronos](https://github.com/MIT-SPARK/Khronos) or [Clio](https://github.com/MIT-SPARK/Clio).

## Usage

`semantic_inference` is not intended for standalone usage.
Instead, the intention is for the launch files in `semantic_inference` to be used in a larger project.
More details about including them can be found in the [closed-set](docs/closed_set.md#using-closed-set-segmentation-online) and [open-set](docs/open_set.md#using-open-set-segmentation-online) documentation.
However, it is possible to do something like
```
ros2 launch semantic_inference_ros closed_set.launch.yaml
```
and then
```
ros2 bag play path/to/rosbag --remap /some/color/image/topic:=/color/image_raw
```
in a separate terminal to quickly test a particular segmentation model. The result can be visualized via
```
ros2 run image_view image_view --ros-args -r image:=/semantic_overlay/image_raw
```

> **Note** </br>
> This usage (remapping the rosbag output topic) is a little bit backwards from how remappings from ROS are normally specified and is because launch files are unable to take remappings from the command line.
