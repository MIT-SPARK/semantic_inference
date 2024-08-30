from semantic_inference.models.segment_refinement import *
from semantic_inference.models.patch_extractor import *
from semantic_inference.models.mask_functions import *
from semantic_inference.models.openset_segmenter import *
from semantic_inference.models.wrappers import *
import torch


def default_device(use_cuda=True):
    """Get default device to use for pytorch."""
    return "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
