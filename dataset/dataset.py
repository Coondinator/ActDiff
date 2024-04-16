import os
import math
import random

from tqdm import tqdm
import numpy as np
import torch
import codecs as cs
from itertools import islice
from torch.utils.data import sampler, Dataset, DataLoader, RandomSampler

class Action_Dataset(Dataset):
    # This dataset is the NeMF latent sequences dataset with text descriptions
    def __init__(self, dataset_path, **kwargs):
        return
    def __len__(self):
        return

    def __getitem__(self, item):
        return