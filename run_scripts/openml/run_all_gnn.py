
import subprocess
import os
import csv
import argparse

import sys
sys.path.append("../../")
from tab_utils import get_dataset_details

parser = argparse.ArgumentParser()
parser.add_argument("--power", type=float, default=2)
args = parser.parse_args()

datasets = [
    "philippine",
    "phoneme",
    "higgs",
    # "wine",
    # "volkert",
    # "christine",
    # "miniboone"
]

for dataset in datasets:

    task = get_dataset_details(dataset)
    subprocess.run([
        "python", 
        f"run_openml_gnn.py", 
        f"--dataset={dataset}",
        f"--power={args.power}",
        f"--n_trials=5"
    ])

    subprocess.run([
        "python", 
        f"run_openml_gnn.py", 
        f"--dataset={dataset}",
        f"--power={args.power}",
        f"--n_trials=5",
        "--add_mask"
    ])