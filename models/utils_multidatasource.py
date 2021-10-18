import torch
import numpy as np;
from torch.autograd import Variable
import time
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import pickle

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda;
        self.P = args.window;
        self.h = args.horizon        
        self.pre_win = args.pre_win; 
        
        #self.m = args.m
        #self.data_org = loadmat(args.data_org_path)['nom_data']
        #self.data_change = loadmat(args.data_change_path)['nom_data']

    def _split(self, data, args):   
        
        n, m = data.shape;  
        train_set = range(self.P+self.h-1, int(args.train * n));
        valid_set = range(int(args.train * n), int((args.train+args.valid) * n));
        test_set = range(int((args.train+args.valid) * n), n);
        train_data = self._batchify(train_set, self.h, data, 0);
        valid_data = self._batchify(valid_set, self.h, data, 0);
        test_data = self._batchify(test_set, self.h, data, 0);
               
        return train_data, valid_data, test_data
        
    
    def _batchify(self, idx_set, horizon, data, label):
        n = len(idx_set);
        X = torch.zeros((n, self.P, self.m));
        if label == 0:
            Label = torch.zeros((n, 1)); 
        else:
            Label = torch.ones((n, 1));  
            
        if self.pre_win == 1:
            Y_pre = torch.zeros((n, self.m));
            Y_long = torch.zeros((n, self.P, self.m));
        else:        
            Y_pre = torch.zeros((n, self.pre_win, self.m)); 
            Y_long = torch.zeros((n, self.P, self.pre_win, self.m));
              
        for i in range(n-self.pre_win+1):
            end_X = idx_set[i];
            start_X = end_X - self.P;     
            np_x = data[start_X:end_X, :]
            X[i,:self.P,:] = torch.from_numpy(np_x);  

            if self.pre_win ==1:
                norm_y = data[idx_set[i], :]
                Y_pre[i,:] = torch.from_numpy(norm_y);
                Y_long[i,:,:] = torch.from_numpy(data[(start_X+self.pre_win) : (end_X+self.pre_win), :])
                
            else:    
                norm_y = data[idx_set[i]:idx_set[i]+self.pre_win, :]
                Y_pre[i,:,:] = torch.from_numpy(norm_y);
                for pre_win_j in range(self.pre_win):
                    Y_long[i,:,pre_win_j,:] = torch.from_numpy(data[(start_X+pre_win_j+1) : (end_X+pre_win_j+1), :])
                
      
        return [X, Label, Y_pre, Y_long];



    def get_batches(self, this_data, batch_size, shuffle = False):
        
        inputs = this_data[0]; 
        labels = this_data[1];
        targets = this_data[2];
        

        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; 
            Label = labels[excerpt];
            Y = targets[excerpt];
            
            if (self.cuda):
                X = X.cuda();
                Label = Label.cuda();
                Y = Y.cuda();
                
            #pdb.set_trace()
            model_inputs = [Variable(X)];

            data = [model_inputs, Variable(Label), Variable(Y)]
            yield data;
#            return data
            start_idx += batch_size


