import numpy as np
import pandas as pd
import h5py
import argparse

parser = argparse.ArgumentParser(prog="Process Data Monolith", usage="python process_data_monolith.py --file ./data/raw/M2_activity.csv")

parser.add_argument("-f", "--file", action="store", required=True, help="File to process", type=str)

args = parser.parse_args()

path = args.file

data = pd.read_csv(path, header=None)

filename = path.split("/")[-1].split(".")[0]

print(data.shape)

trial = np.array([data])

# print(trial.shape)
rearranged_trial = np.einsum("int -> nti", trial)

with h5py.File(f"data/{filename}_monolith.h5", 'w') as f:
    mouse_group = f.create_group(filename.split("_")[0])
    mouse_group.create_dataset('monolith', data=rearranged_trial, compression='gzip', compression_opts=4)