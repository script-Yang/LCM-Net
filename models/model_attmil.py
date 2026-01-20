from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *


class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation
        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Attention_FC_surv, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Constructing Genomic SNN
        if self.fusion is not None:
            hidden = [256, 256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = self.attention_net.to('cuda:0')
            #self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector

        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # return hazards, S, Y_hat, None, None
        return hazards, S, Y_hat