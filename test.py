import pandas as pd
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,TensorDataset
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from skimage.transform import rescale, resize, downscale_local_mean
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import h5py
import pickle
from transformers import *
from sklearn.externals import joblib
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(batch_ndx)