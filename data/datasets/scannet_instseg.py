import json
import os
from pathlib import Path
from copy import deepcopy
import random
import jsonlines
import numpy as np
import torch
from torch_scatter import scatter_mean
from transformers import AutoTokenizer
import volumentations as V
import albumentations as A
import MinkowskiEngine as ME

from data.build import DATASET_REGISTRY
from data.datasets.sceneverse_base import SceneVerseBase
from data.datasets.constant import CLASS_LABELS_200, PromptType
from data.data_utils import make_bce_label
import fpsample

from data.datasets.sceneverse_instseg import SceneVerseInstSeg

@DATASET_REGISTRY.register()
class ScanNetInstSegSceneVerse(SceneVerseInstSeg):
    def __init__(self, cfg, split):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)  
          