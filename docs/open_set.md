# Open-Set Segmentation

## Setting Up

The open-set segmentation interface works with and without ROS. For working with ROS, we assume you have already built your catkin workspace with this repository in it beforehand (i.e., by running `catkin build`).

> **Note </br>**
> If you intend only to use the open-set segmentation interface, you may want to turn off building against TensorRT, which you can do by the following:
> ```
> colcon build --cmake-args --no-warn-unused-cli -DSEMANTIC_INFERENCE_USE_TRT=OFF
> ```

### Installing

We assume you are using a virtual environment. You may want to install `virtualenv` (usually `sudo apt install python3-virtualenv`) if you haven't already.
To set up a virtual environment for use with ROS:
```
python3 -m virtualenv -p /usr/bin/python3 --system-site-packages /desired/path/to/environment
```
Otherwise, omit the ``--system-site-packages`` option:
```
python3 -m virtualenv -p /usr/bin/python3 --download /desired/path/to/environment
```

Then, install `semantic_inference`
```bash
cd /path/to/repo
source PATH_TO_ENVIRONMENT/bin/activate
pip install ./semantic_inference[openset]  # note that the openset extra is required for open-set semantic segmentation
```
You may see dependency version errors from pip if installing into an environment created with `--system-site-packages`. This is expected.
Other open-set segmentation features require other extras (e.g., running with SAM instead of FastSAM requires the `sam` extra, and using dense language feature interpolation requires `f3rm` as an extra).
It is also possible to install via an editable install (i.e., by using `-e` when running `pip install`).

## Models

Note that both CLIP and FastSAM automatically download the relevant model weights when they are first run.
Running with the original SAM may require downloading the model weights. See the official SAM repository [here](https://github.com/facebookresearch/segment-anything) for more details.

## Trying out open-set segmentation nodes

Similar to the example [here](../README.md#using-closed-set-segmentation-online), you can run any of the open-set launch files:

```shell
activate PATH_TO_ENVIRONMENT/bin/activate
## this example just produces an embedding vector per image
# ros2 launch semantic_inference_ros image_embedding_node.launch.yaml
ros2 launch semantic_inference_ros open_set.launch.yaml
```
and then run
```shell
ros2 bag play PATH_TO_BAG --remap INPUT_TOPIC:=/color/image_raw
```

You should see a single embedding vector published under `/semantic/feature` and (if running the full open-set segmenter), the segmentation results under `/semantic/image_raw` and a visualization of the results under `/semantic_color/image_raw` and `/semantic_overlay/image_raw`.

## Using open-set segmentation online

To use the open-set segmentation as part of a larger system, include [open_set.launch.yaml](../semantic_inference_ros/launch/openset.launch.yaml) in your launch file. Often this will look like this:
```yaml
launch:
    # ... rest of launch file ...
    - set_remap: {from: "color/image_raw", to: "YOUR_INPUT_TOPIC_HERE"}
    - include:  {file: "$(find-pkg-share semantic_inference_ros)/launch/opsen_set.launch.yaml"}
```

## Pre-generating semantics

It is also possible to pre-generate semantics when working with recorded data.
To create a rosbag containing the original bag contents *plus* the resulting open-set segmentation, run the following
```
rosrun semantic_inference_ros make_rosbag --copy /path/to/input_bag      \
                                          /color_topic:/output_topic     \
                                          -o /path/to/desired/output_bag
```
replacing `/color_topic` and `/output_topic` with appropriate topic names (usually `/camera_name/color/image_raw` and `/camera_name/semantic/image_raw`).

Additional options exist.
Running without `--copy` will output just the open-set segmentation at the path specified by `-o`.
If no output path is specified, the semantics will be added in-place to the bag after a confirmation prompt (you can disable the prompt with `-y`).
Additional information and documentation is available via `--help`.
