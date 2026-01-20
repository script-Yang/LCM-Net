from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *


class PorpoiseAMIL(nn.Module):
    def __init__(self, size_arg = "small", n_classes=4):
        super(PorpoiseAMIL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        self.classifier = nn.Linear(size[1], n_classes)
        # initialize_weights(self)
                

    def forward(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        # h  = self.classifier(M)
        # return h
        logits = self.classifier(M)
        # print(logits,logits.shape)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # return hazards, S, Y_hat, None, None
        return hazards, S, Y_hat


