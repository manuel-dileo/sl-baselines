import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import add_self_loops

def product_df_to_pyg(df, device='cpu'):
    df.drop(columns=['Code'], inplace=True)
    t = [0, 16, 60, 120, 180, 240, 300, 360, 416]
    ts = [f't{e}' for e in t]
    df.drop(columns=[c for c in df.columns if 't'==c[0] and c not in ts], inplace=True)
    features = [e for e in df.columns if e not in ts]
    x = df[features]
    df['y'] = df.apply(lambda row: row[ts].tolist(), axis=1)
    x_tensor = torch.tensor(x.values).float().to(device)
    y_tensor = torch.tensor(list(df['y'].values)).float().to(device)
    num_nodes = len(df)
    edge_index = torch.Tensor([[],[]])
    edge_index = add_self_loops(edge_index)
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, pos=x_tensor)
    return data

def load_product_df(path):
    df = pd.read_csv(path)
    return df

def split_pyg_data(data):
    split = RandomNodeSplit(num_val=0.20, num_test=0.20)
    data = split(data)
    return data

def load_dataset(path, device):
    df = load_product_df(path)
    data = product_df_to_pyg(df, device)
    data = split_pyg_data(data)
    return data

if __name__ == '__main__':
    df = pd.read_csv('data/dataset_h')
    data = product_df_to_pyg(df)
    print(data)