#!/usr/bin/env python3

import subprocess
import time


def modify_ensemble_weights(weight_A, weight_B):
    # Read the original file
    with open('./ensemble_train.py', 'r') as file:
        lines = file.readlines()

    # Modify the lines
    for i in range(len(lines)):
        if 'weight_A =' in lines[i]:
            lines[i] = f"    weight_A = {weight_A}  # Weight for candidate 1 (Trump)\n"
        elif 'weight_B =' in lines[i]:
            lines[i] = f"    weight_B = {weight_B}  # Weight for candidate 2 (Harris)\n"
        elif 'political = "-political-ensemble' in lines[i]:
            lines[i] = f'    political = "-political-ensemble-{int(weight_A*100)}-Donald-{int(weight_B*100)}-Kamala"\n'

    # Write the modified content back to the file
    with open('./ensemble_train.py', 'w') as file:
        file.writelines(lines)


def run_experiment(weight_A, weight_B):
    print(f"\n{'='*50}")
    print(
        f"Starting experiment with weights: Trump={weight_A:.2f}, Harris={weight_B:.2f}")
    print(f"{'='*50}\n")

    # Modify the weights
    modify_ensemble_weights(weight_A, weight_B)
    print(f"Modified weights: Trump={weight_A:.2f}, Harris={weight_B:.2f}")

    # Run the training
    train_cmd = "python ./ensemble_train.py"
    print(f"Running: {train_cmd}")
    subprocess.run(train_cmd, shell=True)

    # Wait a bit between experiments
    time.sleep(5)


if __name__ == "__main__":
    # List of weight pairs to try (Trump weight, Harris weight)
    experiments = [
        (0.6, 0.4),  # Original weights
        (0.7, 0.3),  # More weight to Trump
        (0.5, 0.5),  # Equal weights
        (0.8, 0.2),  # Much more weight to Trump
        (0.4, 0.6),  # More weight to Kamala
        (0.2, 0.8),  # Much more weight to Kamala
    ]

    for i, (weight_A, weight_B) in enumerate(experiments, 1):
        print(f"\nRunning Experiment {i} of {len(experiments)}")
        run_experiment(weight_A, weight_B)

    print("\nAll experiments completed!")
