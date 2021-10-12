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
pip install onnx onnxruntime torch torchvision seaborn
pip install git+https://github.com/CSAILVision/semantic-segmentation-python.git@master
```
