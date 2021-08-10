# Semantic Recolor Utilities

Installation requires Cuda and TensorRT:

To install a minimal setup (after adding the Cuda repositories):
```
sudo apt install cuda-librarires-11-1 cuda-libraries-dev-11-1 cuda-nvrtc-dev-11-1 \
                 libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
```

For the scripts:

```
python3 -m venv semantic_recolor  # or whatever you want
source semantic_recolor/bin/activate
pip install --upgrade pip
pip install wheel
pip install onnx onnxruntime torch torchvision seaborn
pip install git+https://github.com/CSAILVision/semantic-segmentation-python.git@master
```
