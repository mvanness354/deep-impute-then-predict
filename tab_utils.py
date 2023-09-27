import numpy as np
import pandas as pd
from math import sqrt, log, pi
from sklearn.preprocessing import scale, LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from scipy import stats


def get_dataset_details(dataset_name):
    if dataset_name in ["phoneme", "christine", "arcene", "higgs", "miniboone", "dating", "bands", "voting", "arrhythmia", "philippine"]:
        task = "binary"
    elif dataset_name in ["volkert", "wine", "dilbert", "eucalyptus", "college", "mice"]:
        task = "multiclass"
    elif dataset_name in ["space", "tecator", "housing", "yolanda", "elevator", "crime", "meta"]:
        task = "regression"
    else:
        raise "Unknown dataset"
    
    return task

def load_openml_dataset(dataset_name):
    dataset_ids = {
        "phoneme": 1489,
        "miniboone": 41150,
        "wine": 40498,
        "higgs": 23512,
        "christine": 41142,
        "volkert": 41166,
        "dilbert": 41163,
        "housing": 537,
        "elevator": 216,
        "yolanda": 42705,
        "tecator": 505,
        "space": 507,
        "arcene": 1458,
        "eucalyptus": 188,
        "crime": 315,
        "college": 488,
        "meta": 566,
        "dating": 40536,
        "bands": 6332,
        "voting": 56,
        "arrhythmia": 1017,
        "philippine": 41145,
        "mice": 40966,
    }

    id = dataset_ids[dataset_name]
    task = get_dataset_details(dataset_name)

    if task == "regression":
        data = fetch_openml(data_id=id)
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = (y - y.mean()) / y.std()

        if dataset_name in ["crime", "meta"]:
            # X = X[[c for c in X.columns if X[c].dtype.name != "category"]]
            X = X[[c for c in X.columns if X[c].dtype.name != "category" and X[c].isna().sum() > 0]]

    
    else:
        data = fetch_openml(data_id=id)
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)

        if dataset_name in ["volkert", "christine", "arcene"]:
            X = X[[c for c in X.columns if X[c].nunique() > 2]] # Remove unnecessary features
        elif dataset_name == "higgs":
            X = X.iloc[:-1, :] # last columns has NAs
            y = y[:-1]
        elif dataset_name in ["arrhythmia"]:
            X = X[[c for c in X.columns if X[c].dtype.name != "category" and X[c].isna().sum() == 0]]
        elif dataset_name in ["eucalyptus", "college", "mice", "dating", "bands"]:
            X = X[[c for c in X.columns if X[c].dtype.name != "category" and X[c].isna().sum() > 0]]

    return X, y

def simple_mask(X, p=0.5, rng=None, seed=None, return_na=False):
    if not rng and not seed:
        rng = np.random.default_rng()
    elif not rng:
        rng = np.random.default_rng(seed)

    # Simple MCAR mask
    mask = rng.binomial(n=1, p=p, size=X.shape)

    if return_na:
        mask = np.where(mask == 0, np.nan, mask)

    return mask

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MNAR_mask(x, side="tail", rng=None, seed=None, power=1, standardize=False, return_na=False):
    if not rng and not seed:
        rng = np.random.default_rng()
    elif not rng:
        rng = np.random.default_rng(seed)

    if standardize:
        x = StandardScaler().fit_transform(x)

    if side == "tail":
        probs = sigmoid((np.abs(x) - 0.75)*power)
    elif side == "mid":
        probs = sigmoid((-np.abs(x) + 0.75)*power)
    elif side == "left":
        probs = sigmoid(-x*power)
    elif side == "right":
        probs = sigmoid(x*power)
    else:
        raise ValueError(f"Side must be one of tail, mid, left, or right, got {side}")

    mask = rng.binomial(1, probs, size=x.shape)
    
    if return_na:
        mask = np.where(mask == 0, np.nan, mask)
    
    return probs, mask

def make_classifier(n, p, k, seed=10, cov=None, power=2, noise_scale=2):
    rng = np.random.default_rng(seed)

    if cov is not None:
        X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    else:

        if p > k:
            A = rng.normal(0, 1, size=(k, k))
            cov = A.T @ A
            X = rng.multivariate_normal(mean=rng.normal(size=k), cov=cov, size=n)

            n_extra_feats = p - k
            W = rng.normal(size=(k, n_extra_feats))
            X = np.column_stack([X, X @ W])
        else:
            A = rng.normal(0, 1, size=(p // 2, p))
            cov = A.T @ A
            X = rng.multivariate_normal(mean=rng.normal(size=p), cov=cov, size=n)

    beta = rng.normal(size=p)
    logits = X @ beta + rng.normal(loc=0, scale=noise_scale, size=n)
    logits = scale(logits.reshape(-1, 1)).reshape(-1)
    y = rng.binomial(n=1, p = 1 / (1 + np.exp(-power*logits)), size=n)

    return X, y

def make_regression(n, p, k=None, rank=None, seed=10, cov=None):
    rng = np.random.default_rng(seed)

    if cov is not None:
        X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    else:
        if k is not None:
            if rank is None:
                rank = k // 2

            A = rng.normal(0, 1, size=(k, rank))
            cov = A @ A.T + np.diag(rng.uniform(low=0.01, high=0.1, size=k))
            X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

            n_extra_feats = p - k
            W = rng.normal(size=(k, n_extra_feats))
            new_feats = X @ W

            new_feats_noise_level = np.var(new_feats, axis=0) / 10

            new_feats += rng.multivariate_normal(mean=np.zeros(p - k), cov=np.diag(new_feats_noise_level))

            X = np.column_stack([X, new_feats])
        else:
            if rank is None:
                rank = p // 2

            A = rng.normal(0, 1, size=(rank, p))
            cov = A.T @ A + np.diag(rng.uniform(low=0.01, high=0.1, size=p))
            X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    beta = rng.normal(size=p)
    y = X @ beta

    # var_y = np.mean((current_y - np.mean(current_y))**2)
    # sigma2_noise = var_y/snr

    # noise = rng.normal(
        # loc=0, scale=sqrt(sigma2_noise), size=n_samples-current_size)
    # current_y += noise

    # noise_var = np.var(y) / 10
    # noise = rng.normal(loc=0, scale=np.sqrt(noise_var), size=n)
    # y += noise

    # Scale y
    y = (y - y.mean()) / y.std()

    return X, y



def make_neumiss_regression(
    n_samples=10000, 
    n_features=10,
    link="linear", 
    snr=10,
    prop_latent=0.3,
    seed=10
):

    rng = np.random.default_rng(seed)

    # Generate mean and cov
    mean = np.zeros(n_features) + rng.standard_normal(n_features)
    B = rng.standard_normal((n_features, int(prop_latent*n_features)))
    cov = B.dot(B.T) + np.diag(
        rng.uniform(low=0.01, high=0.1, size=n_features)
    )
    
    # Generate beta
    beta = np.repeat(1., n_features + 1)
    var = beta[1:].dot(cov).dot(beta[1:])
    beta[1:] *= 1/sqrt(var)

    X = np.empty((0, n_features))
    y = np.empty((0, ))
    current_size = 0

    current_X = rng.multivariate_normal(
            mean=mean, cov=cov,
            size=n_samples-current_size,
            check_valid='raise')

    dot_product = current_X.dot(beta[1:]) + beta[0]

    if link == 'linear':
        current_y = dot_product
    elif link == 'square':
        curvature = 1
        current_y = curvature*(dot_product-1)**2
    elif link == 'cube':
        current_y = beta[0] + curvature*dot_product**3 - 3*dot_product
    elif link == 'stairs':
        curvature = 20
        current_y = dot_product - 1
        for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
            tmp = np.sqrt(np.pi/8)*curvature*(dot_product + b)
            current_y += a*stats.norm.cdf(tmp)
    elif link == 'discontinuous_linear':
        current_y = dot_product + (dot_product > 1)*3

    var_y = np.mean((current_y - np.mean(current_y))**2)
    sigma2_noise = var_y/snr

    noise = rng.normal(
        loc=0, scale=sqrt(sigma2_noise), size=n_samples-current_size)
    current_y += noise

    # current_M = np.zeros((n_samples-current_size, n_features))
    # for j in range(n_features):
    #     X_j = current_X[:, j]
    #     if sm_type == 'probit':
    #         lam = sm_params['lambda']
    #         c = sm_params['c'][j]
    #         prob = stats.norm.cdf(lam*X_j - c)
    #     elif sm_type == 'gaussian':
    #         k = sm_params['k']
    #         sigma2_tilde = sm_params['sigma2_tilde'][j]
    #         mu_tilde = mean[j] + k*sqrt(cov[j, j])
    #         prob = np.exp(-0.5*(X_j - mu_tilde)**2/sigma2_tilde)

    #     current_M[:, j] = rng.binomial(n=1, p=prob, size=len(X_j))

    # if not perm:
    #     np.putmask(current_X, current_M, np.nan)
    # else:
    #     for j in range(n_features):
    #         new_j = perms[j]
    #         np.putmask(current_X[:, new_j], current_M[:, j], np.nan)

    X = np.vstack((X, current_X))
    y = np.hstack((y, current_y))

    current_size = n_samples

    return X, y