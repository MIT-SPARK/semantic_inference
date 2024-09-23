# Open-Set Segmentation

## Setting Up

The open-set segmentation interface works with and without ROS. For working with ROS, we assume you have already built your catkin workspace with this repository in it beforehand (i.e., by running `catkin build`).

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

> :warning: **Warning** <br>
> Note that newer versions of `setuptools` are incompatible with `--system-site-packages` on Ubuntu 20.04. Do not use `--download` and `--system-site-packages` and expect the installation to work (specifically with external packages specified by git url).

Then, install `semantic_inference`
```bash
cd /path/to/repo
source /path/to/environment
pip install ./semantic_inference[openset]  # note that the openset extra is required for open-set semantic segmentation
```
You may see dependency version errors from pip if installing into an environment create with `--system-site-packages`. This is expected.
Other open-set segmentation features require other extras (e.g., running with SAM instead of FastSAM requires the `sam` extra, and using dense language feature interpolation requires `f3rm` as an extra).
It is also possible to install via an editable install (i.e., by using `-e` when running `pip install`).

### Models

Note that both CLIP and FastSAM automatically download the relevant model weights when they are first run.
Running with the original SAM may require downloading the model weights. See the official SAM repository [here](https://github.com/facebookresearch/segment-anything) for more details.
