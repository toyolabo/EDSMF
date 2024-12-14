#!/usr/bin/env python3

import subprocess
import time

def run_experiment(fea_num, data_path):
    print(f"\n{'='*50}")
    print(f"Starting experiment with fea_num={fea_num} and data_path={data_path}")
    print(f"{'='*50}\n")

    # Modify the train.py file
    modify_cmd = f"python ./modify_train.py {fea_num} {data_path}"
    print(f"Running: {modify_cmd}")
    subprocess.run(modify_cmd, shell=True)

    # Run the training
    train_cmd = "python ./train.py"
    print(f"Running: {train_cmd}")
    subprocess.run(train_cmd, shell=True)

    # Wait a bit between experiments
    time.sleep(5)

if __name__ == "__main__":
    experiments = [
        # (fea_num, data_path)
        (8, '../dataset/SP500/reduced_sp500_2024.npy'),
        (11, '../dataset/SP500/reduced_sp500_2024_political.npy'),
        (8, '../dataset/SP500/reduced_sp500_2024.npy'),
        (11, '../dataset/SP500/reduced_sp500_2024_political.npy'),
        (8, '../dataset/SP500/reduced_sp500_2024.npy'),
        (11, '../dataset/SP500/reduced_sp500_2024_political.npy'),
    ]

    for i, (fea_num, data_path) in enumerate(experiments, 1):
        print(f"\nRunning Experiment {i} of {len(experiments)}")
        run_experiment(fea_num, data_path)

    print("\nAll experiments completed!")