import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import argparse
import yaml
import itertools
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append("../../models") 
from tf_models import MLPClassifier, NeuMissMLP, MLPMIWAE, MLPNotMIWAE, MLPRegressor
sys.path.append("../../")
from tab_utils import load_openml_dataset, simple_mask, MNAR_mask, make_regression, make_neumiss_regression
from tf_utils import get_tf_dataset

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=5)
parser.add_argument("--n", type=int, default=10000)
parser.add_argument("--p", type=float, default=20)
args = parser.parse_args()

def get_score_from_noise(latent_dim, seed):

    print()
    print("RUNNING latent_dim", latent_dim, "seed", seed)
    print()

    n = int(args.n)
    p = int(args.p)
    # cov = np.ones(shape=(p, p)) * sigma
    # np.fill_diagonal(cov, 1)
    # # X, y = make_classifier(n=n, p=p, seed=seed, k=p)
    # X, y = make_regression(n=n, p=p, seed=seed, cov=cov)

    X, y = make_neumiss_regression(n_samples=n, n_features=p, prop_latent=latent_dim, seed=seed)

    # Generate MCAR mask
    mask = simple_mask(X, seed=seed, return_na=True, p=0.3)
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
    preprocessor = make_pipeline(
        StandardScaler(),
    )
    train_X = preprocessor.fit_transform(train_X).clip(min=-10, max=10)
    val_X = preprocessor.transform(val_X).clip(min=-10, max=10)
    test_X = preprocessor.transform(test_X).clip(min=-10, max=10)
    test_X_complete = preprocessor.transform(test_X_complete).clip(min=-10, max=10)

    train_mask = np.isnan(train_X)
    test_mask = np.isnan(test_X)

    # First train model without imputation
    tf.random.set_seed(seed)
    np.random.seed(seed)

    batch_size = 128
    train_dataset = get_tf_dataset([train_X, train_y], batch_size)
    val_dataset = get_tf_dataset([val_X, val_y], batch_size, shuffle=False)
    test_dataset = get_tf_dataset([test_X, test_y], batch_size, shuffle=False)

    model = MLPRegressor(n_layers=4)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    checkpoint_path = "checkpoints/"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_rmse', mode="min", min_delta=1e-4, patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_rmse',
            mode='min',
            save_best_only=True,
        )
    ]
    model.fit(
        train_dataset,
        epochs=100,
        validation_data = val_dataset,
        callbacks = callbacks,
    )

    model.load_weights(checkpoint_path)
    outputs = model.evaluate(test_dataset)
    test_rmse_no_imputation = dict(zip(model.metrics_names, outputs))["rmse"]

    # Get imputation rmse with 0 imputation
    imputation_rmse_no_imputation = mean_squared_error(np.zeros_like(test_X_complete[test_mask]), test_X_complete[test_mask], squared=False)

    train_input = train_X.copy()

    # imputation via missforest
    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(),
    )
    train_input = imputer.fit_transform(train_input)  

    val_input = val_X.copy()
    val_input = imputer.transform(val_input) 

    test_input = test_X.copy()
    test_input = imputer.transform(test_input)

    # Get imputation rmse
    imputation_rmse_mf = mean_squared_error(test_input[test_mask], test_X_complete[test_mask], squared=False)

    # Train MLP model
    tf.random.set_seed(seed)
    np.random.seed(seed)

    batch_size = 128
    train_dataset = get_tf_dataset([train_input, train_y], batch_size)
    val_dataset = get_tf_dataset([val_input, val_y], batch_size, shuffle=False)
    test_dataset = get_tf_dataset([test_input, test_y], batch_size, shuffle=False)

    model = MLPRegressor(n_layers=4)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    checkpoint_path = "checkpoints/"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_rmse', mode="min", min_delta=1e-4, patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_rmse',
            mode='min',
            save_best_only=True,
        )
    ]
    model.fit(
        train_dataset,
        epochs=100,
        validation_data = val_dataset,
        callbacks = callbacks,
    )

    model.load_weights(checkpoint_path)
    outputs = model.evaluate(test_dataset)
    test_rmse_imputation = dict(zip(model.metrics_names, outputs))["rmse"]


    return test_rmse_no_imputation, test_rmse_imputation, imputation_rmse_no_imputation, imputation_rmse_mf

from itertools import product
n_trials = args.n_trials
seeds = np.arange(10, 10 + n_trials)
results = [
    [sigma] + list(get_score_from_noise(sigma, seed)) 
    for sigma, seed in product(np.arange(0.05, 0.95, 0.05), seeds)
]

results_df = pd.DataFrame(results, columns=["sigma", "rmse_zero_impute", "rmse_mf_impute", "imputation_rmse_zero_impute", "imputation_rmse_mf_impute"])
print(results_df)
results_df.to_csv(f"../results/synthetic_score_by_latent_dim.csv", index=False)
