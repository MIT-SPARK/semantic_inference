# Exporting Instructions

When exporting a new model, you will also need to write a config similar to [this](../semantic_inference_ros/config/models/ade20k-efficientvit_seg_l2.yaml).

## ADE20k baseline models

You can export all models (that are compatible) from the MIT scene parsing challenge via [this script](../exporting/export_mit_semseg.py).
At the moment, only `hrnetv2-c1` and `mobilenetv2dilated-c1_deepsup` are compatible.
This may change as the newer onnx export method in torch becomes stable (or not, it is unclear whether or not the custom batch norm operator will ever work with the export).
To run the script, you will want to create a separate virtual environment and install dependencies lists in the [pyproject file](../semantic_inference/pyproject.toml).

## EfficientViT

Make a separate virtual environment (with `python3.10`) and just install efficientvit to the environment (pip installing from the git repo url worked well).
Run [this script](../exporting/export_efficientvit.py). You may need other dependencies (`click`, `matplotlib`, etc.).
