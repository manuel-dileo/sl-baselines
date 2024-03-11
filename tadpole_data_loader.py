import torch
import pickle
import numpy as np
import os.path as osp
import torch
from torch_geometric.data import Data


class TadpoleDataset(torch.utils.data.Dataset):
    """Class to load the Tadpole dataset"""

    def __init__(self, device='cpu', fold=0, full=True):
        with open('data/tadpole_data.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        
        if not full: # For DGM original paper they use modality 1 (M1) for both node representation and graph learning. 
                     #See the original dataset
            X_ = X_[...,:30,:] 
        
        self.n_features = X_.shape[-2]
        self.num_classes = y_.shape[-2]
        
        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.argmax(torch.from_numpy(y_[:,:,fold]).float(),dim=1).to(device)
        self.train_mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        self.test_mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X,self.y,self.mask, [[]]
    
    def get_pyg_data(self):
        edge_index = torch.Tensor([[],[]])
        data = Data(x = self.X, edge_index = edge_index, y = self.y, train_mask = self.train_mask,\
                    val_mask = self.test_mask, test_mask = self.test_mask)
        return data
    
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = TadpoleDataset(device='cpu')
    data = loader.get_pyg_data()
    print(data)
        