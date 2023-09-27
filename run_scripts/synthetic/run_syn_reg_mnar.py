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
from tf_models import MLPClassifier, NeuMissMLP, MLPMIWAE, MLPNotMIWAE, MLPRegressor, AutoEncodePredictor
sys.path.append("../../")
from utils import load_openml_dataset, simple_mask, MNAR_mask, get_tf_dataset, make_classifier, make_regression

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=5)
parser.add_argument("--power", type=float, default=0.5)
# parser.add_argument("--cov", type=float, default=0.8)
args = parser.parse_args()

def run_tf(model_name, seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask):

    if model_name == "MLP":
        model = MLPRegressor(n_layers=4)
    elif model_name == "SupMIWAE":
        model = MLPMIWAE(n_input=p, n_layers=4, n_samples=50, regression=True)
    elif model_name == "NeuMiss":
        model = NeuMissMLP(n_input=p, neumiss_depth=10, mlp_n_layers=4, impute=True, regression=True)
    elif model_name == "ae":
        model = AutoEncodePredictor(n_input=p, n_layers=4, regression=True)
    else:
        raise ValueError(f"Model name must be one of MLP, SupMIWAE, NeuMiss, or ae, got {model_name}")


    tf.random.set_seed(seed)
    np.random.seed(seed)

    batch_size = 128
    train_dataset = get_tf_dataset([train_X, train_y], batch_size)
    val_dataset = get_tf_dataset([val_X, val_y], batch_size, shuffle=False)
    test_dataset = get_tf_dataset([test_X, test_y], batch_size, shuffle=False)

    if model_name in ["SupMIWAE", "SupNotMIWAE"]:
        model.compile(
            optimizer = "adam",
            metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
    else:
        model.compile(
            optimizer = "adam",
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

    # Get prediction scores
    model.load_weights(checkpoint_path)
    outputs = model.evaluate(test_dataset)
    test_score = dict(zip(model.metrics_names, outputs))["rmse"]

    # Get imputation scores
    test_X_imputed = tf.concat([
        model.impute(batch_X) for batch_X, _ in test_dataset
    ], axis=0).numpy()
    imputation_rmse = mean_squared_error(test_X_complete[test_mask], test_X_imputed[test_mask], squared=False)

    return [seed, test_score, imputation_rmse]

results = []

seeds = np.arange(10, 10 + args.n_trials)
for seed in seeds:

    n = 10000
    p = 10
    cov = np.ones(shape=(p, p)) * 0.7
    np.fill_diagonal(cov, 1)

    # prop_latent = 0.3
    # rng = np.random.default_rng(seed)
    # B = rng.standard_normal((p, int(prop_latent*p)))
    # cov = B.dot(B.T) + np.diag(
    #     rng.uniform(low=0.01, high=0.1, size=p)
    # )

    X, y = make_regression(n=n, p=p, seed=10, cov=cov)

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
    preprocessor = make_pipeline(
        StandardScaler(),
    )
    train_X = preprocessor.fit_transform(train_X).clip(min=-10, max=10)
    val_X = preprocessor.transform(val_X).clip(min=-10, max=10)
    test_X = preprocessor.transform(test_X).clip(min=-10, max=10)
    test_X_complete = preprocessor.transform(test_X_complete).clip(min=-10, max=10)
    test_mask = np.isnan(test_X)

    # First do imputation using missforest
    mf = IterativeImputer(estimator=HistGradientBoostingRegressor(random_state=seed), random_state=seed)
    print("FITTING MISSFOREST")
    train_X_imputed_mf = mf.fit_transform(train_X)
    val_X_imputed_mf = mf.transform(val_X)
    test_X_imputed_mf = mf.transform(test_X)

    results.append(
        [args.power, "MLP + Missforest"] + run_tf(
            "MLP", seed, train_X_imputed_mf, train_y, val_X_imputed_mf, val_y, test_X_imputed_mf, test_y, test_X_complete, test_mask
        )
    )

    # Now for the other models as well
    results.append(
        [args.power, "MLP"] + run_tf(
            "MLP", seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask
        )
    )
    results.append(
        [args.power, "SupMIWAE"] + run_tf(
            "SupMIWAE", seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask
        )
    )
    results.append(
        [args.power, "NeuMiss"] + run_tf(
            "NeuMiss", seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask
        )
    )

results = pd.DataFrame(results, columns=["Power", "Model", "Seed", "Test_RMSE", "Imputation_RMSE"])
print(results)
results.to_csv(f"../../results/synthetic_reg_mnar_power{int(args.power)}_small.csv", index=False)
