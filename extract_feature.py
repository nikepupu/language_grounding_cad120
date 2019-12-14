#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
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

parser = argparse.ArgumentParser(description='CAD120 feature extraction')
parser.add_argument('-v', '--video-max-length', default=400, type=int,
                     dest='video_length',
                    help='maximum length of a video, measured by the number of frames')

parser.add_argument('-c', '--caption-max-length', default=15, type=int,
                    dest = 'sentence_length',
                    help='maximum length of a video, measured by the number of frames')

args = parser.parse_args()

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
        
subject = 'subject5'

cad120 = CAD120Dataset('./caption_'+ subject +'.csv', transform)


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
caption_words = []

video_length = args.video_length
sentence_length = args.sentence_length

for i in tqdm(range(l)):
    sample = cad120[i]
   
    
    if cap != sample['caption']:
        if len(block) != 0:

            while len(block) < video_length:
                block = [torch.zeros(1,8192).cuda()] + block


            block = torch.stack(block)
            features.append(block.cpu())
            block = []

        tmp_cap = sample['caption'].strip() 
        l = len(tmp_cap .split())
        #print(l)
        #print(tmp_cap)
        if l < sentence_length:
            for j in range(sentence_length-l):
                tmp_cap = '[PAD]' + tmp_cap

        #print(tmp_cap)

        input_ids = torch.tensor([tokenizer.encode(tmp_cap, add_special_tokens=False)])

        with torch.no_grad():
            res = embedding_model(input_ids)[0]
            res = res.cpu()
        
        assert res.shape[1] == sentence_length, print(tmp_cap, res.shape, input_ids)
        caption.append(res)

        caption_words.append(input_ids.cpu())
            
    cap = sample['caption']
    img =  sample['image'].unsqueeze(0).cuda()
    a = res50_conv(img)
    a = a.reshape([-1,512*4*4])
    

   
    
    block.append(a)

#features.append(np.array(block))
while len(block) < video_length:
    block = [torch.zeros(1,8192).cuda()] + block
block = torch.stack(block)
features.append(block)
l = len(features)
features = np.array(features)



filename = 'caption_words_' + subject + '.bin' 
joblib.dump(caption_words,filename)

#caption = caption
filename = 'caption_' + subject + '.bin'
joblib.dump(caption, filename) 

#features = features
filename = 'features_' + subject + '.bin'
joblib.dump(features, filename) 