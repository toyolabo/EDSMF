#!/usr/bin/env python3

import sys

def modify_train_file(fea_num, data_path):
    # Read the original file
    with open('./train.py', 'r') as file:
        lines = file.readlines()

    # Modify the lines
    for i in range(len(lines)):
        # Update fea_num
        if lines[i].strip().startswith('fea_num ='):
            lines[i] = f'fea_num = {fea_num} # Number of features in consideration\n'
        # Update data path
        elif 'data = np.load(' in lines[i] and '/SP500/' in lines[i]:
            lines[i] = f'    data = np.load(\'{data_path}\')\n'

    # Write the modified content back to the file
    with open('./train.py', 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_train.py <fea_num> <data_path>")
        print("Example: python modify_train.py 8 '../dataset/SP500/reduced_sp500_2024.npy'")
        sys.exit(1)

    fea_num = int(sys.argv[1])
    data_path = sys.argv[2]
    
    modify_train_file(fea_num, data_path)
    print(f"Modified train.py with fea_num={fea_num} and data_path={data_path}")