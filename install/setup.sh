#!/bin/bash

mkdir -p $HOME/.semantic_inference/

if [[ ! -e $HOME/.semantic_inference/ade20k-efficientvit_seg_l2.onnx ]]; then
    echo "Downloading 'ade20k-efficientvit_seg_l2.onnx'..."
    pipx run gdown -q --fuzzy https://drive.google.com/file/d/1XRcsyLSvqqhqNIaOI_vmqpUpmBT6gk9-/view?usp=drive_link --output $HOME/.semantic_inference/ade20k-efficientvit_seg_l2.onnx
fi

if [[ ! -e $HOME/.semantic_inference/ade20k-hrnetv2-c1.onnx ]]; then
    echo "Downloading 'ade20k-hrnetv2-c1.onnx'..."
    pipx run gdown -q --fuzzy https://drive.google.com/file/d/1vwTKs5g-xrY_2z_V3_TFnmqF0QYd24OM/view?usp=drive_link --output $HOME/.semantic_inference/ade20k-hrnetv2-c1.onnx
fi

if [[ ! -e $HOME/.semantic_inference/ade20k-mobilenetv2dilated-c1_deepsup.onnx ]]; then
    echo "Downloading 'ade20k-mobilenetv2dilated-c1_deepsup.onnx'..."
    pipx run gdown -q --fuzzy https://drive.google.com/file/d/1siTG4Pce9o6iRKtKYZDEPWsru4LwNclo/view?usp=drive_link --output $HOME/.semantic_inference/ade20k-mobilenetv2dilated-c1_deepsup.onnx
fi
