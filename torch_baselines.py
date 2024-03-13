from torch_geometric.nn import GAT, GCN, TransformerConv, DynamicEdgeConv, MLP, Linear
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import erdos_renyi_graph
import torch

class MLPRegressor(torch.nn.Module):
    """
        MLPRegressor for tabular data
    """
    def __init__(self, channel_list):

        super(MLPRegressor, self).__init__()
        self.mlp = MLP(channel_list)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, data: Data):
        x = data.x
        h = self.mlp(x)
        return h
class KnnGNN(torch.nn.Module):
    """
        Pre-processed the graph with Knn for graph construction:
        Use a KnnGraph for graph construction, then GCN/GAT to solve node-regression
    """
    def __init__(self, in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels=1,
                 model='GCN',
                 k=7):
        super(KnnGNN, self).__init__()
        if model == 'GCN':
            self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_channels,\
                           num_layers = num_layers, out_channels= out_channels)
        elif model == 'GAT':
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels,\
                           num_layers = num_layers, out_channels= out_channels, v2=False)
        else:
            raise ValueError(f"{model} model not available")
        self.graph_construction = KNNGraph(k=k)
    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data: Data):
        data = self.graph_construction(data)
        x, edge_index = data.x, data.edge_index
        h = self.gnn(x, edge_index)
        return h

class GraphTransformer(torch.nn.Module):
    """
        Attention-based model for graph-construction:
        Use a TransformerConv to process a complete graph,
        then a KnnGraph on the processed embeddings for graph construction,
        finally a GCN to solve node-regression
    """
    def __init__(self, in_channels,
                 hidden_channels,
                 num_layers,
                 attn_head = 1,
                 out_channels=1,
                 k=7,
                 device='cpu'):
        super(GraphTransformer, self).__init__()
        self.transformer = TransformerConv(in_channels=in_channels, out_channels=hidden_channels,\
                                           heads=attn_head, concat=False)
        self.gnn = GCN(in_channels=hidden_channels, hidden_channels=hidden_channels,\
                       num_layers = num_layers, out_channels= out_channels)
        self.graph_construction = KNNGraph(k=k)
        self.device = device
        self.k = k

    def reset_parameters(self):
        self.transformer.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data: Data):
        num_nodes = data.num_nodes
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=1.0, directed=False).to(self.device) #Generate a complete-graph
        x = data.x
        h = self.transformer(x, edge_index)
        attn_data = KNNGraph(k=self.k)(Data(x=h, pos=h)) #Obtain KnnGraph from an attention-based approach
        x, edge_index = attn_data.x.to(self.device), attn_data.edge_index.to(self.device)
        h = self.gnn(x, edge_index)
        return h

class DGCNN(torch.nn.Module):
    """
        A model using the dynamic edge convolutional operator from the DGCNN(“Dynamic Graph CNN for Learning on Point Clouds”) paper
        where the graph is dynamically constructed using nearest neighbors in the feature space.
    """
    def __init__(self, input_channels,
                hidden_channels,
                output_channels=1,
                k=7):

        super(DGCNN, self).__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * input_channels, hidden_channels]), k)
        self.conv2 = DynamicEdgeConv(MLP([2 * hidden_channels, hidden_channels]), k)
        self.decoder = Linear(hidden_channels, output_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data: Data):
        x = data.x
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.decoder(h)
        return h

