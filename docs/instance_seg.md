# Instance Segmentation

## Setting Up

It is recommended to setup instance segmentation with ROS, and we assume you have already built your workspace with this repository in it beforehand (i.e., by running `colcon build`).

### Installing

We assume you are using a virtual environment. You may want to install `virtualenv` (usually `sudo apt install python3-virtualenv`) if you haven't already.
To set up a virtual environment for use with ROS:
```shell
python3 -m virtualenv -p /usr/bin/python3 --system-site-packages <DESIRED_PATH_TO_ENVIRONMENT>
```
Otherwise, omit the ``--system-site-packages`` option:
```shell
python3 -m virtualenv -p /usr/bin/python3 --download <DESIRED_PATH_TO_ENVIRONMENT>
```

> NOTE: we default the virtual environment name to `gdsam2` currently.

Then, install `semantic_inference`
```shell
cd <PATH_TO_REPO>
source <PATH_TO_ENVIRONMENT>/bin/activate
pip install ./semantic_inference
```

The above setup allows you to use `yolov11`, in order to use `Grounded Sam 2`, we have to manually install it.
```shell
# cd to your favorite path, we can default to `~/.semantic_inference/`
git clone -b more_gpu https://github.com/MultyXu/Grounded-SAM-2.git
```
And follow the `README.md` in the cloned repo to install gdsam2.

## Setup models
For `Grounded Sam 2`, put (or symlink) `GroundingDINO_SwinT_OGC.py` under `~/.semantic_inference/gdsam2_config/`. And, put `sam2.1_hiera_large.pt` and `groundingdino_swint_ogc.pth` under `~/.semantic_inference/`

For `Yolo`, download [yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) under `~/.semantic_inference/`

## Trying out close-set instance segmentation nodes

Similar to the example [here](../README.md#usage), you can run any of the instance segmentation launch files:

```shell
activate <PATH_TO_ENVIRONMENT>/bin/activate

ros2 launch semantic_inference_ros instance_segmentation.launch.yaml
```
and then run
```shell
ros2 bag play PATH_TO_BAG --remap INPUT_TOPIC:=/color/image_raw
```

You should see raw instance segmentation result published under `/semantic/feature` a visualization of the results under `/semantic_overlay/image_raw`.
> **IMPORTANT: the raw instance segmentation result is a 32 bit int with first 16 bit representing the semantic id and the last 16 bit for instance id. See [instance_segmenter.py](../semantic_inference/python/semantic_inference/models/instance_segmenter.py) for more information.**


## Using close-set instance segmentation online

To use the close-set instance segmentation as part of a larger system, include [instance_segmentation.launch.yaml](../semantic_inference_ros/launch/instance_segmentation.launch.yaml) in your launch file. Often this will look like this:
```yaml
launch:
    # ... rest of launch file ...
    - set_remap: {from: "color/image_raw", to: "YOUR_INPUT_TOPIC_HERE"}
    - include:  {file: "$(find-pkg-share semantic_inference_ros)/launch/instance_segmentation.launch.yaml"}
```
