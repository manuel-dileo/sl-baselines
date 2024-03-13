from torch_baselines import MLPRegressor, KnnGNN, GraphTransformer, DGCNN
from data_loader import load_dataset
from train_eval import train,test
import random
import torch
import numpy as np
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()


# Add command-line options
parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--dataset', type=str, help='Dataset name, stored in the data folder')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight Decay')
parser.add_argument('--multiple_seeds', '--flag', action='store_true', help='Run baselines over five seeds')

args = parser.parse_args()

def run_all_baselines(args):
    seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    data = load_dataset(f'data/{args.dataset}', device)
    lr = args.lr
    weight_decay = args.weight_decay

    num_features = data.x.size(1)
    num_properties = data.y.size(1)
    mlp = MLPRegressor([num_features,128, 84, 64, num_properties]).to(device)
    mlp_optimizer = torch.optim.Adam(params=mlp.parameters(), lr=lr, weight_decay = weight_decay)
    mlp.reset_parameters()

    gcn = KnnGNN(in_channels=num_features, hidden_channels=84, num_layers=1,\
                 out_channels=num_properties, model='GCN').to(device)
    gcn_optimizer = torch.optim.Adam(params=gcn.parameters(), lr=lr, weight_decay=weight_decay)
    gcn.reset_parameters()

    gat = KnnGNN(in_channels=num_features, hidden_channels=84, num_layers=1,\
                 out_channels=num_properties, model='GAT').to(device)
    gat_optimizer = torch.optim.Adam(params=gat.parameters(), lr=lr, weight_decay=weight_decay)
    gat.reset_parameters()

    gt = GraphTransformer(in_channels=num_features, hidden_channels=84, num_layers=1,\
                          out_channels=num_properties, attn_head=8, device=device).to(device)
    gt_optimizer = torch.optim.Adam(params=gt.parameters(), lr=lr, weight_decay=weight_decay)
    gt.reset_parameters()

    dgcnn = DGCNN(input_channels=num_features, hidden_channels=84,\
                  output_channels=num_properties).to(device)
    dgcnn_optimizer = torch.optim.Adam(params=dgcnn.parameters(), lr=lr, weight_decay=weight_decay)
    dgcnn.reset_parameters()

    criterion = torch.nn.L1Loss()

    mlp = train(model=mlp, data=data, optimizer=mlp_optimizer, criterion=criterion)
    gcn = train(model=gcn, data=data, optimizer=gcn_optimizer, criterion=criterion)
    gat = train(model=gat, data=data, optimizer=gat_optimizer, criterion=criterion)
    gt = train(model=gt, data=data, optimizer=gt_optimizer, criterion=criterion)
    dgcnn = train(model=dgcnn, data=data, optimizer=dgcnn_optimizer, criterion=criterion)

    mlp_mae, mlp_rmse = test(mlp, data, data.test_mask)
    gcn_mae, gcn_rmse = test(gcn, data, data.test_mask)
    gat_mae, gat_rmse = test(gat, data, data.test_mask)
    gt_mae, gt_rmse = test(gt, data, data.test_mask)
    dgcnn_mae, dgcnn_rmse = test(dgcnn, data, data.test_mask)

    print("Results on test set")
    print(f'Seed {seed}')
    print(f"MLP: MAE: {mlp_mae}, RMSE: {mlp_rmse}")
    print(f"GCN: MAE: {gcn_mae}, RMSE: {gcn_rmse}")
    print(f"GAT: MAE: {gat_mae}, RMSE: {gat_rmse}")
    print(f"GraphTransf: MAE: {gt_mae}, RMSE: {gt_rmse}")
    print(f"DGCNN: MAE: {dgcnn_mae}, RMSE: {dgcnn_rmse}")

    return mlp_mae, mlp_rmse,gcn_mae, gcn_rmse, gat_mae, gat_rmse,\
        gt_mae, gt_rmse, dgcnn_mae, dgcnn_rmse

if __name__=='__main__':
    if args.multiple_seeds:
        mlp_maes = []
        mlp_rmses = []
        gcn_maes = []
        gcn_rmses = []
        gat_maes = []
        gat_rmses = []
        gt_maes = []
        gt_rmses = []
        dgcnn_maes = []
        dgcnn_rmses = []
        for s in range(5):
            args.seed = s
            mlp_mae, mlp_rmse, gcn_mae, gcn_rmse, gat_mae, gat_rmse,\
                gt_mae, gt_rmse, dgcnn_mae, dgcnn_rmse = run_all_baselines(args)
            mlp_maes.append(mlp_mae)
            mlp_rmses.append(mlp_rmse)
            gcn_maes.append(gcn_mae)
            gcn_rmses.append(gcn_rmse)
            gat_maes.append(gat_mae)
            gat_rmses.append(gat_rmse)
            gt_maes.append(gt_mae)
            gt_rmses.append(gt_rmse)
            dgcnn_maes.append(dgcnn_mae)
            dgcnn_rmses.append(dgcnn_rmse)
        print("MLP")
        print(f"\t RMSE: {np.mean(mlp_rmses)} +- {np.std(mlp_rmses)}")
        print(f"\t MAE: {np.mean(mlp_maes)} +- {np.std(mlp_maes)}")
        print()
        print("GCN")
        print(f"\t RMSE: {np.mean(gcn_rmses)} +- {np.std(gcn_rmses)}")
        print(f"\t MAE: {np.mean(gcn_maes)} +- {np.std(gcn_maes)}")
        print()
        print("GAT")
        print(f"\t RMSE: {np.mean(gat_rmses)} +- {np.std(gat_rmses)}")
        print(f"\t MAE: {np.mean(gat_maes)} +- {np.std(gat_maes)}")
        print()
        print("Graph-Transformer")
        print(f"\t RMSE: {np.mean(gt_rmses)} +- {np.std(gt_rmses)}")
        print(f"\t MAE: {np.mean(gt_maes)} +- {np.std(gt_maes)}")
        print()
        print("DGCNN")
        print(f"\t RMSE: {np.mean(dgcnn_rmses)} +- {np.std(dgcnn_rmses)}")
        print(f"\t MAE: {np.mean(dgcnn_maes)} +- {np.std(dgcnn_maes)}")
        print()

    else:
        run_all_baselines(args)
