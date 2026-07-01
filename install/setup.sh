#!/bin/bash

# Installs dependencies for building and running semantic_inference
# Usage: ./setup.sh [--no-models] [--no-closed-set] [--no-cuda] [--no-python]"
#
# This installs a minimal set of dependencies that *should* work for TensorRT, but YMMV.
# Note: CUDA is hard-coded to install the 24.04 keyring and repos.
#
# Options:
#   --no-models: Do not download model weights
#   --no-closed-set: Do not install closed-set dependencies
#   --no-cuda: Do not install the CUDA keyring (if you already have CUDA set up)
#   --no-python: Do not set up python virtual environment for instance and open-set segmentation
#   -h/--help: Show brief usage information

function download_model() {
    if [[ ! -e $HOME/.semantic_inference/$1 ]]; then
        echo "Downloading '$1'..."
        wget "$2" -O "$HOME"/.semantic_inference/"$1"
    fi
}


# from https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODELS=true  # Download models
CLOSED_SET=true  # Install closed-set deps
CUDA=true  # Install cuda key
MAKE_ENV=true  # Make python environment
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-models)
            MODELS=false
            ;;
        --no-closed-set)
            CLOSED_SET=false
            ;;
        --no-cuda)
            CUDA=false
            ;;
        --no-python)
            MAKE_ENV=false
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [--no-models] [--no-closed-set] [--no-cuda] [--no-python]"
            exit 0
    esac
    shift
done

if [ "$CLOSED_SET" = true ]; then
    if [ "$CUDA" = true ]; then
        # TODO(nathan) grab actual release info to pick correct platform
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
    fi

    sudo apt update
    version_info=$(apt-cache policy libnvinfer-dev)
    regex=" +Candidate: +[0-9.\-]+\+cuda([0-9]+)\.([0-9]+)"
    if [[ $version_info =~ $regex ]]; then
        version="${BASH_REMATCH[1]}"
        minor_version="${BASH_REMATCH[2]}"
    else
        echo "Could not find CUDA version!"
        exit 1
    fi

    sudo apt install libnvinfer-dev libnvonnxparsers-dev libnvinfer-plugin-dev "cuda-nvcc-${version}-${minor_version}" -y
fi

if [ "$MODELS" = true ]; then
    mkdir -p "$HOME"/.semantic_inference/
    download_model ade20k-efficientvit_seg_l2.onnx "https://www.dropbox.com/scl/fi/qtaqm3htsdjlnoyqjvrol/ade20k-efficientvit_seg_l2.onnx?rlkey=3evg4gfeybd0wie0gom8535zo&st=1ocu1vl7&dl=1"
    download_model ade20k-hrnetv2-c1.onnx "https://www.dropbox.com/scl/fi/6hvyfgtk1j0uqt1t7dal9/ade20k-hrnetv2-c1.onnx?rlkey=k009j60g556h9p79bdoxgskbx&st=sqz7sqyl&dl=1"
    download_model ade20k-mobilenetv2dilated-c1_deepsup.onnx "https://www.dropbox.com/scl/fi/dlcizojblgaq8dnhd94o4/ade20k-mobilenetv2dilated-c1_deepsup.onnx?rlkey=5m7tv9x8bt0gsg1q77tert9ic&st=ixu66280&dl=1"
fi


if [ "$MAKE_ENV" = true ]; then
    PACKAGE_PATH="$(dirname ${SCRIPT_DIR})"
    mkdir -p "$HOME"/.semantic_inference
    python3 -m venv --system-site-packages --clear "$HOME"/.semantic_inference/env
    source "$HOME"/.semantic_inference/env/bin/activate
    python3 -m pip install -e "${PACKAGE_PATH}/semantic_inference"[openset]
    deactivate
fi
