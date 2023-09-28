import os
import numpy as np
import pandas as pd
import argparse
import yaml
import itertools
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

import sys
sys.path.append("../../")
from tab_utils import load_openml_dataset, simple_mask, MNAR_mask, get_dataset_details

sys.path.append("../../GRAPE")
from training.gnn_y import train_gnn_y
from uci.uci_subparser import add_uci_subparser
from uci.uci_data import get_data
from utils.utils import auto_select_gpu
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=5)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--power", type=float, default=2.0)
parser.add_argument("--add_mask", action="store_true")
args = parser.parse_args()

# select device
if torch.cuda.is_available():
    cuda = auto_select_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    device = torch.device('cuda:{}'.format(cuda))
else:
    print('Using CPU')
    device = torch.device('cpu')

def train_epoch(task, epoch, data, model, impute_model, predict_model, opt, scheduler, add_mask=False):
    n_row, n_col = data.df_X.shape
    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    train_y_mask = all_train_y_mask.clone().detach()

    model.train()
    impute_model.train()
    predict_model.train()

    known = 0.7
    known_mask = get_known_mask(known, int(train_edge_attr.shape[0] / 2)).to(device)
    double_known_mask = torch.cat((known_mask, known_mask), dim=0)
    
    known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

    opt.zero_grad()
    x_embd = model(x, known_edge_attr, known_edge_index)
    X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
    X = torch.reshape(X, [n_row, n_col])

    if add_mask:
        mask = torch.tensor(np.isnan(data.df_X).to_numpy().astype(int)).to(device)
        X = torch.concatenate((X, mask), dim=1)

    pred = predict_model(X)[:, 0]
    pred_train = pred[train_y_mask]
    label_train = y[train_y_mask]

    if task == "regression":
        pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]
        loss = F.mse_loss(pred_train, label_train)
    elif task == "binary":
        pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]
        loss = F.binary_cross_entropy(pred_train, label_train)
    elif task == "multiclass":
        pred = predict_model(X)
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]
        loss = F.nll_loss(pred_train, label_train)
    loss.backward()
    opt.step()
    train_loss = loss.item()
    if scheduler is not None:
        scheduler.step(epoch)

    return train_loss

def val_epoch(task, data, model, impute_model, predict_model, add_mask=False):
    n_row, n_col = data.df_X.shape
    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    train_y_mask = all_train_y_mask.clone().detach()

    model.eval()
    impute_model.eval()
    predict_model.eval()
    with torch.no_grad():
        x_embd = model(x, train_edge_attr, train_edge_index)
        X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
        X = torch.reshape(X, [n_row, n_col])

        if add_mask:
            mask = torch.tensor(np.isnan(data.df_X).to_numpy().astype(int)).to(device)
            X = torch.concatenate((X, mask), dim=1)

        if task == "regression":
            pred = predict_model(X)[:, 0]
            pred_train = pred[train_y_mask]
            label_train = y[train_y_mask]
            val_score = mean_squared_error(label_train.cpu().numpy(), pred_train.cpu().numpy(), squared=False)
        elif task == "binary":
            pred = predict_model(X)[:, 0]
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            val_score = roc_auc_score(label_test.cpu().numpy(), torch.sigmoid(pred_test).cpu().numpy())
        elif task == "multiclass":
            pred = predict_model(X)
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            val_score = accuracy_score(label_test.cpu().numpy(), torch.argmax(pred_test, dim=1).cpu().numpy())
        else:
            raise ValueError("Task must be one of: regression, binary, multiclass")
        
    return val_score

def test_epoch(task, data, model, impute_model, predict_model, test_X_complete, add_mask=False):
    n_row, n_col = data.df_X.shape
    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    train_y_mask = all_train_y_mask.clone().detach()

    model.eval()
    impute_model.eval()
    predict_model.eval()
    with torch.no_grad():
        x_embd = model(x, train_edge_attr, train_edge_index)
        X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
        X = torch.reshape(X, [n_row, n_col])

        # Get imputation RMSE using sklearn
        missing_mask = np.isnan(data.df_X)
        imputation_rmse = mean_squared_error(
            X.cpu().numpy()[missing_mask], 
            test_X_complete.to_numpy()[missing_mask], 
            squared=False
        )

        if add_mask:
            mask = torch.tensor(np.isnan(data.df_X).to_numpy().astype(int)).to(device)
            X = torch.concatenate((X, mask), dim=1)

        if task == "regression":
            pred = predict_model(X)[:, 0]
            pred_train = pred[test_y_mask]
            label_train = y[test_y_mask]
            test_score = mean_squared_error(label_train.cpu().numpy(), pred_train.cpu().numpy(), squared=False)
        elif task == "binary":
            pred = predict_model(X)[:, 0]
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            test_score = roc_auc_score(label_test.cpu().numpy(), torch.sigmoid(pred_test).cpu().numpy())
        elif task == "multiclass":
            pred = predict_model(X)
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            test_score = accuracy_score(label_test.cpu().numpy(), torch.argmax(pred_test, dim=1).cpu().numpy())
        else:
            raise ValueError("Task must be one of: regression, binary, multiclass")
        
    return test_score, imputation_rmse


# Turns a dictionary into a class
class Dict2Class(object): 
    def __init__(self, my_dict): 
        for key in my_dict:
            setattr(self, key, my_dict[key])

def run_gnn(seed, task, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask):

    # # Read in defeault task params
    # with open("../../models/model_params.yml", 'r') as f:
    #     model_params = yaml.safe_load(f)["gnn"]

    # Read in default grape params
    with open("../../models/grape_config.yaml", 'r') as f:
        grape_args = Dict2Class(yaml.safe_load(f))

    

    train_X_gnn = pd.concat([train_X, val_X])
    train_y_gnn = pd.concat([train_y, val_y])
    val_mask = np.array([False] * len(train_y) + [True] * len(val_y))

    print("Getting data")
    start_time = time.time()
    train_data = get_data(
        train_X_gnn, train_y_gnn.to_numpy(), grape_args.node_mode, grape_args.train_edge, grape_args.split_sample, grape_args.split_by, 
        grape_args.train_y, grape_args.seed, has_missing=True, normalize=True, test_mask=val_mask
    )
    print(f"Time to get data: {time.time() - start_time}")
    
    model = get_gnn(train_data, grape_args).to(device)
    n_row, n_col = train_data.df_X.shape

    if grape_args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, grape_args.impute_hiddens.split('_')))

    if grape_args.concat_states:
        input_dim = grape_args.node_dim * len(model.convs) * 2
    else:
        input_dim = grape_args.node_dim * 2

    impute_model = MLPNet(input_dim, 1,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=grape_args.impute_activation,
                            dropout=grape_args.dropout).to(device)

    if grape_args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, grape_args.predict_hiddens.split('_')))

    if args.add_mask:
        pred_input_size = n_col * 2
    else:
        pred_input_size = n_col

    if task == "regression":
        predict_model = MLPNet(pred_input_size, 1,
                            hidden_layer_sizes=predict_hiddens,
                            dropout=grape_args.dropout, output_activation=None).to(device)
    elif task == "binary":
        predict_model = MLPNet(pred_input_size, 1,
                            hidden_layer_sizes=predict_hiddens,
                            dropout=grape_args.dropout, output_activation="sigmoid").to(device)
    elif task == "multiclass":
        predict_model = MLPNet(pred_input_size, 1,
                            hidden_layer_sizes=predict_hiddens,
                            dropout=grape_args.dropout, output_activation="softmax").to(device)
    else:
        raise ValueError("Task must be one of: regression, binary, multiclass")


    trainable_parameters = list(model.parameters()) \
                            + list(impute_model.parameters()) \
                            + list(predict_model.parameters())

    # build optimizer
    scheduler, opt = build_optimizer(grape_args, trainable_parameters)


    n_epochs = 1000

    if task == "regression":
        best_score = np.inf
    else:
        best_score = 0

    min_delta = 1e-4
    patience = 50
    epochs_without_improvement = 0
    for epoch in range(n_epochs):
        train_loss = train_epoch(task, epoch, train_data, model, impute_model, predict_model, opt, scheduler, add_mask=args.add_mask)
        val_score = val_epoch(task, train_data, model, impute_model, predict_model, add_mask=args.add_mask)
        print(f"Epoch {epoch} | Train Loss: {np.round(train_loss, 4)} | Val Score : {np.round(val_score, 4)}")

        if task == "regression":
            if val_score < best_score - min_delta:
                best_score = val_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            if val_score > best_score + min_delta:
                best_score = val_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement > patience:
            break

    test_mask = np.array([True] * len(test_y))
    test_data = get_data(
        test_X, test_y.to_numpy(), grape_args.node_mode, grape_args.train_edge, grape_args.split_sample, grape_args.split_by, 
        grape_args.train_y, grape_args.seed, has_missing=True, normalize=True, test_mask=test_mask
    )

    test_score, imputation_rmse = test_epoch(task, test_data, model, impute_model, predict_model, test_X_complete, add_mask=args.add_mask)
    return [seed, test_score, imputation_rmse]
    

    


# Read in openml data
X, y = load_openml_dataset(args.dataset)
y = pd.Series(y)

dataset_task = get_dataset_details(args.dataset)

results = []

seeds = np.arange(10, 10 + args.n_trials)
for seed in seeds:

    # Generate mask
    _, mask = MNAR_mask(X, side="right", power=args.power, seed=seed, return_na=True, standardize=True)
    X_masked = X * mask

    # Train/Val/Test split of 60/20/20
    train_X, test_X, train_y, test_y, train_X_complete, test_X_complete = train_test_split(
        X_masked, 
        y, 
        X,
        test_size=0.20, 
        random_state=seed
    )
    train_X, val_X, train_y, val_y, train_X_complete, val_X_complete = train_test_split(
        train_X, 
        train_y, 
        train_X_complete,
        test_size=0.25,
        random_state=seed,
    )

    # Prepare dataset
    preprocessor = StandardScaler().set_output(transform="pandas")
    train_X = preprocessor.fit_transform(train_X).clip(lower=-10, upper=10)
    val_X = preprocessor.transform(val_X).clip(lower=-10, upper=10)
    test_X = preprocessor.transform(test_X).clip(lower=-10, upper=10)
    test_X_complete = preprocessor.transform(test_X_complete).clip(lower=-10, upper=10)
    test_mask = np.isnan(test_X)


    print()
    print(f"Running {args.dataset} with seed {seed}")
    print()
    results.append(run_gnn(
        seed, dataset_task, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask
    ))


score_df = pd.DataFrame(results, columns=["seed", "score", "impute_score"])
score_df.to_csv(f"../../results/openml/{args.dataset}_power{int(args.power)}_gnn_mask{args.add_mask}.csv", index=False)
    
    

