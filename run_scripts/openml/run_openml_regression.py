import os
import numpy as np
import pandas as pd
import argparse
import yaml
import itertools
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../../models") 
from tf_models import MLPRegressor, NeuMissMLP, MLPMIWAE, MLPNotMIWAE
sys.path.append("../../")
from tab_utils import load_openml_dataset, simple_mask, MNAR_mask
from tf_utils import get_tf_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# ----------- Parse Args ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=5)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--power", type=float, default=2.0)
args = parser.parse_args()

# Model Map
# Model Map
model_map = {
    "mlp": MLPRegressor,
    "mlp_mim": MLPRegressor,
    "neumiss": NeuMissMLP,
    "neumiss_mim": NeuMissMLP,
    "supmiwae": MLPMIWAE,
    "supnotmiwae": MLPNotMIWAE,
    "gbt": HistGradientBoostingRegressor,
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
        score = mean_squared_error(test_y, model.predict(test_X), squared=False)
        
        return [seed, model_name, score, None]
        
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        batch_size = 128
        train_dataset = get_tf_dataset([train_X, train_y], batch_size)
        val_dataset = get_tf_dataset([val_X, val_y], batch_size, shuffle=False)
        test_dataset = get_tf_dataset([test_X, test_y], batch_size, shuffle=False)

        if model_name not in ["mlp", "mlp_mim"]:
            model_params["n_input"] = X.shape[1]
            model_params["regression"] = True
            
        model = model_map[model_name](**model_params)

        if model_name in ["supmiwae", "supnotmiwae"]:
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
        
        model.load_weights(checkpoint_path)

        if "miwae" in model_name:
            test_y_pred = tf.concat([
                model(batch_X)[0] for batch_X, _ in test_dataset
            ], axis=0).numpy()
            score = mean_squared_error(test_y, test_y_pred, squared=False)
        else:
            test_y_pred = tf.concat([
                model(batch_X) for batch_X, _ in test_dataset
            ], axis=0).numpy()
            score = mean_squared_error(test_y, test_y_pred, squared=False)



        # Get imputation scores
        test_X_imputed = tf.concat([
            model.impute(batch_X) for batch_X, _ in test_dataset
        ], axis=0).numpy()
        test_mask = np.isnan(test_X)
        imputation_rmse = mean_squared_error(test_X_complete[test_mask], test_X_imputed[test_mask], squared=False)
        
        return [seed, model_name, score, imputation_rmse]


# Read in openml data
X, y = load_openml_dataset(args.dataset)

results = []

seeds = np.arange(10, 10 + args.n_trials)
models = ["mlp", "mlp_mim", "neumiss", "neumiss_mim", "supmiwae", "supnotmiwae", "gbt"]
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
    preprocessor = make_pipeline(
        StandardScaler(),
    )
    train_X = preprocessor.fit_transform(train_X).clip(min=-10, max=10)
    val_X = preprocessor.transform(val_X).clip(min=-10, max=10)
    test_X = preprocessor.transform(test_X).clip(min=-10, max=10)
    test_X_complete = preprocessor.transform(test_X_complete).clip(min=-10, max=10)
    test_mask = np.isnan(test_X)

    for model in models:
        print()
        print(f"Running {model} on {args.dataset} with seed {seed}")
        print()
        results.append(run_tf(
            model, seed, train_X, train_y, val_X, val_y, test_X, test_y, test_X_complete, test_mask
        ))


score_df = pd.DataFrame(results, columns=["seed", "model", "score", "impute_score"])
score_df.to_csv(f"../../results/openml/{args.dataset}_power{int(args.power)}.csv", index=False)
    
    

