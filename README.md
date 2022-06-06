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

For the scripts:

```
python3 -m venv semantic_recolor  # or whatever you want
source semantic_recolor/bin/activate
pip install --upgrade pip
pip install wheel
pip install onnx onnxruntime torch torchvision seaborn pyyaml
pip install git+https://github.com/CSAILVision/semantic-segmentation-python.git@master
```

## Models

You can export a model via [this script](scripts/export_onnx_model.py).

Several pre-exported models live [here](https://drive.google.com/drive/folders/1GrmgFDFCssDxKe_Nyx8PPTK1pRMA0gEO?usp=sharing)

## New Datasets

To adapt to a new dataset (or new set of labels), you will have to:

  - Modify [this script](scripts/make_ade150k_color_config.py) to change the label groupings (Kimera-Semantics is limited to 20 colors but can be modified [here](https://github.mit.edu/SPARK/Kimera-Semantics/blob/aaf0a89cb8d921ec3a72e74559d6ec197cbdd825/kimera_semantics/include/kimera_semantics/common.h#L26))
  - Run the above modified script to produce a new configuration
  - Export a model and write a config similar to [this](config/hrnetv2_360_640_v12.yaml). A good naming scheme is `{model_name}_{image_height}_{image_width}_{onnx_instruction_version}.yaml`. To check if the model is valid and show input/output names, run [this](scripts/check_onnx_model.py) script.
  - Pass in the appropriate arguments to the launch file (most likely `model_name` and `dataset_name`
