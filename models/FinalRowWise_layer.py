import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.preprocessing import normalize
import math
## granger full rank

class FR_Model(nn.Module):
    def __init__(self, args, p_len):
        super(FR_Model, self).__init__()
        self.pre_win = args.pre_win
        self.y_dim = args.y_dim       
        self.weight = nn.Parameter(torch.ones([self.y_dim, p_len, self.pre_win]))
        self.weight = nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(torch.ones([self.y_dim, self.pre_win])) 
      
    def forward(self, x):       
        if self.pre_win ==1:
            final_y = torch.empty(x.shape[0], self.y_dim) 
        else :
            final_y = torch.empty(x.shape[0], self.pre_win, self.y_dim) 
        for j in range(self.y_dim):           
            if self.pre_win ==1:   
                final_y[:,j] = F.linear(x[:,j,:], self.weight[j,:].view(1, self.weight.shape[1]), self.bias[j,:]).view(-1);               
            else:
                final_y[:,:,j] = F.linear(x[:,j,:], self.weight[j,:].transpose(1,0), self.bias[j,:]);               
        return final_y;
    