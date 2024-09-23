# Semantic Inference

This repository provides code for running inference on images to provide both closed and open-set semantics.
The repository is currently split into two pieces:
  - Inference using dense 2D closed-set semantic segmentation models is implemented in c++ using TensorRT
  - Inference using open-set segmentation models and language features is implemented in python

Both pieces of the repository have a ROS interface associated with them, split between c++ and python as appropriate.

## Getting started

The recommended use-case for the repository is with ROS.
We assume you have cloned this repository into your catkin workspace and run rosdep to get any missing dependencies.  This usually looks like the following:
```bash
cd /path/to/catkin_ws/src
git clone git@github.com:MIT-SPARK/semantic_inference.git
rosdep install --from-paths . --ignore-src -r -y
```
We also assume some familiarity with working in ROS. A quick primer for setting up a minimal workspace is [below](#making-a-workspace).

Follow one (or both) of the following setup-guides as necessary:
- [Closed-Set](docs/closed_set.md#setting-up)
- [Open-Set](docs/open_set.md#setting-up)

## Running online

The primary use-case for semantic inference is to be run as part of larger system that requires dense semantic information.

## Running closed-set segmentation

TBD

## Running open-set segmentation

TBD

### Running offline

It is also possible to make rosbags containing segmentation results.

## Creating closed-set segmentation rosbags

TBD

## Creating open-set segmentation rosbags

To create a rosbag containing the original bag contents *plus* the resulting open-set segmentation (the normal desired output), run the following
```
rosrun semantic_inference_ros make_rosbag --copy /path/to/input_bag /color_topic:/output_topic -o /path/to/desired/output_bag
```
replacing `/color_topic` and `/output_topic` with appropriate topic names (usually `/camera_name/color/image_raw` and `/camera_name/semantic/image_raw`).

Additional options exist.
Running without `--copy` will output just the open-set segmentation at the path specified by `-o`.
If no output path is specified, the semantics will be added in-place to the bag after a confirmation prompt (you can disable the prompt with `-y`).
Additional information and documentation is available via `--help`.

## Making a workspace

```bash
# Initialize necessary tools for working with ROS and catkin
sudo apt install python3-catkin-tools python3-rosdep
sudo rosdep init
rosdep update

# Setup the workspace
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release
```
