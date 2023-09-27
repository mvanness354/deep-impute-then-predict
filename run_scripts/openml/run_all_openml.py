
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
    "volkert",
    "miniboone",
    "christine",
]

for dataset in datasets:

    task = get_dataset_details(dataset)
    subprocess.run([
        "python", 
        f"run_openml_{task}.py", 
        f"--dataset={dataset}",
        f"--power={args.power}"
    ])