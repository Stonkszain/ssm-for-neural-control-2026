import numpy as np
import pandas as pd
import h5py
import argparse

parser = argparse.ArgumentParser(prog="Process Data Pseudotrials", usage="python process_data_pseudotrials.py --file ./data/raw/M1_activity.csv")

parser.add_argument("-f", "--file", action="store", required=True, help="File to process", type=str)

args = parser.parse_args()

path = args.file

data = pd.read_csv(path, header=None)

print(data.shape[1])

filename = path.split("/")[-1].split(".")[0]

total_timesteps = data.shape[1]
timesteps = list(range(0, total_timesteps, 401))[:-1] + list(range(200, total_timesteps, 401))[:-1]

trials = np.array([data.iloc[:, timestep:timestep+400].to_numpy() for timestep in timesteps])

rearranged_trials = np.einsum('int -> nti', trials)

with h5py.File(f'data/{filename}.h5', 'w') as f:
    mouse_group = f.create_group(filename.split("_")[0])
    mouse_group.create_dataset('all', data=rearranged_trials, compression='gzip', compression_opts=4)