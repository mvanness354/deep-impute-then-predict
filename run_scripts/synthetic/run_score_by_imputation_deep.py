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
from tab_utils import load_openml_dataset, simple_mask, MNAR_mask, make_regression, make_neumiss_regression
from tf_utils import get_tf_dataset

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=5)
parser.add_argument("--n", type=int, default=10000)
parser.add_argument("--p", type=float, default=20)
args = parser.parse_args()

model_map = {
    "mlp": MLPRegressor,
    "mlp_mim": MLPRegressor,
    "neumiss": NeuMissMLP,
    "neumiss_mim": NeuMissMLP,
    "supmiwae": MLPMIWAE,
    "supnotmiwae": MLPNotMIWAE,
    "ae": AutoEncodePredictor,
    # "gbt": HistGradientBoostingClassifier,
}


def run_tf(model_name, seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask):

    # Read in defeault task params
    with open("../../models/model_params.yml", 'r') as f:
        model_params = yaml.safe_load(f)[model_name]


    if model_name == "gbt":
        np.random.seed(seed)
        model_params["random_state"] = seed
        
        model = model_map[model_name](**model_params)
        model.fit(train_X, train_y)
        score = roc_auc_score(test_y, model.predict_proba(test_X)[:, 1])
        
        return [seed, model_name, score, None]
        
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        batch_size = 128
        train_dataset = get_tf_dataset([train_X, train_y], batch_size)
        val_dataset = get_tf_dataset([val_X, val_y], batch_size, shuffle=False)
        test_dataset = get_tf_dataset([test_X, test_y], batch_size, shuffle=False)

        if model_name not in ["mlp", "mlp_mim"]:
            model_params["n_input"] = train_X.shape[1]
            model_params["regression"] = True

        model = model_map[model_name](**model_params)

        if model_name in ["supmiwae", "supnotmiwae"]:
            model.compile(
                optimizer = "adam",
                metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
            )
        else:

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
        score = dict(zip(model.metrics_names, outputs))["rmse"]



        # Get imputation scores
        test_X_imputed = tf.concat([
            model.impute(batch_X) for batch_X, _ in test_dataset
        ], axis=0).numpy()
        test_mask = np.isnan(test_X)
        imputation_rmse = mean_squared_error(test_X_complete[test_mask], test_X_imputed[test_mask], squared=False)
        
        return [seed, score, imputation_rmse]

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
    mask = simple_mask(X, seed=seed, return_na=True, p=0.5)
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

    models = ["mlp", "neumiss", "supmiwae"]

    # First train model without imputation
    tf.random.set_seed(seed)
    np.random.seed(seed)

    results = []
    for model in models:
        print()
        print(f"Running {model} with seed {seed} and latent_dim {latent_dim}")
        print()
        results.append([latent_dim, model] + run_tf(
            model, seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask
        ))

    return results

from itertools import product
n_trials = args.n_trials
seeds = np.arange(11, 11 + n_trials)
results = []
for sigma, seed in product(np.arange(0.05, 1, 0.05), seeds):
    results += get_score_from_noise(sigma, seed)

# results = get_score_from_noise(0.3, 10)

results_df = pd.DataFrame(results, columns=["latent_dim", "model", "seed", "score", "impute_score"])
print(results_df)
results_df.to_csv(f"../../results/synthetic_score_by_latent_dim_deep_3.csv", index=False)
