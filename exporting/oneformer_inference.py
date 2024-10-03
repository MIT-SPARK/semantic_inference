# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Run inference with oneformer."""
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import numpy as np
import pathlib
import click
import torch
import random
import zmq
import sys


def _setup_cfg(oneformer_path, config_file, weights_file):
    sys.path.insert(1, str(oneformer_path))
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from oneformer import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
        add_convnext_config,
    )

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(config_file)
    defaults = [
        "MODEL.IS_TRAIN",
        "False",
        "MODEL.IS_DEMO",
        "True",
        "MODEL.WEIGHTS",
        weights_file,
    ]
    cfg.merge_from_list(defaults)
    cfg.freeze()
    return cfg


@click.command()
@click.argument("oneformer_path", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("weights_file", type=click.Path(exists=True))
@click.option("-p", "--port", type=int, default=5555)
@click.option("-u", "--url", type=str, default="127.0.0.1")
def main(oneformer_path, config_file, weights_file, port, url):
    """Run script."""
    oneformer_path = pathlib.Path(oneformer_path).expanduser().absolute()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{url}:{port}")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = _setup_cfg(oneformer_path, config_file, weights_file)

    cpu_device = torch.device("cpu")
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    msg = "= Inference Server Started!"
    print("=" * 80)
    print(msg + (" " * (79 - len(msg))) + "=")
    print("=" * 80)

    with torch.no_grad():
        while True:
            img = socket.recv_pyobj()
            height, width = img.shape[:2]
            image = aug.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            task = "The task is semantic"
            inputs = {"image": image, "height": height, "width": width, "task": task}
            labels = model([inputs])[0]["sem_seg"].argmax(dim=0).to(cpu_device).numpy()
            socket.send_pyobj(labels)


if __name__ == "__main__":
    main()
