import subprocess

datasets = [
    "phoneme",
    "philippine",
    "higgs",
    "wine",
]

powers = [2, 0]

for power in powers:
    for dataset in datasets:
        subprocess.run([
            "python", 
            f"run_all_openml.py",  
            f"--dataset={dataset}",
            f"--power={power}"
        ])