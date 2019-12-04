#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
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
from transformers import *
import pickle
from sklearn.externals import joblib



normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std= [0.229, 0.224, 0.225])

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128,128)),
		transforms.ToTensor(),
		normalize,
		])


class CAD120Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_read = pd.read_csv(csv_file, header=None)
        self.files = list(self.csv_read[0])
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.files[idx]
        
        image = io.imread(img_name)

       
        if self.transform:
            image = self.transform(image)
            
        
        caption = self.csv_read[1][idx]

        sample = {'image':  image, 'caption': caption}

            
        return sample
        

cad120 = CAD120Dataset('./caption_subject3.csv', transform)


res50_model = models.resnet34(pretrained=True)
res50_model.cuda()
res50_model.eval()



res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
for p in res50_conv.parameters():
    p.requires_grad = False
res50_conv.cuda()
res50_conv.eval()



l = cad120.__len__()
#features = np.zeros((l, 2048,8,8), dtype=np.float32)
features = []
caption = []
model_class, tokenizer_class, pretrained_weights = (BertModel,BertTokenizer,'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
embedding_model = model_class.from_pretrained(pretrained_weights)
cap = ""
block = []

for i in tqdm(range(l)):
    sample = cad120[i]
   
    
    if cap != sample['caption']:
        if len(block) != 0:
            block = torch.stack(block)
            features.append(block)
            block = []
        input_ids = torch.tensor([tokenizer.encode(sample['caption'], add_special_tokens=True)])
        with torch.no_grad():
            res = embedding_model(input_ids)[0]
            
        caption.append(res)
            
    cap = sample['caption']
    img =  sample['image'].unsqueeze(0).cuda()
    a = res50_conv(img)

    a = a.reshape([-1,512*4*4])
    block.append(a)

#features.append(np.array(block))
block = torch.stack(block)
features.append(block)
l = len(features)
features = np.array(features)

   



filename = 'caption_subject3.bin'
joblib.dump(caption, filename) 

filename = 'features_subject3.bin'
joblib.dump(features, filename) 