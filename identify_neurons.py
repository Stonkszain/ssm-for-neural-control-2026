import os
import pickle
import sys
sys.path.insert(0, "/ssm-1")
sys.modules.pop('ssm', None)
sys.modules.pop('ssm.util', None)

import copy
import h5py
from deltaf import findBaselineF0

import scipy.io as sio
import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(12345)
from sklearn.decomposition import FactorAnalysis

import ssm

import argparse

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog = "Identify Neurons", usage = "python identify_neurons.py --mouse M1 --activity eat --iterations 1000")

parser.add_argument("-f", "--file", action="store", required=True, help="Pickle file path", type=str)
parser.add_argument("-d", "--data", action="store", required=False, default=None, help="Path of data to plot trajectory", type=str)
parser.add_argument("-n", "--top-n", action="store", required=False, default=200, help="Top N neurons", type=int)
parser.add_argument("-f2", "--file2", action="store", required=False, default="", help="Pickle file being compared with path", type=str)

args = parser.parse_args()

# print(args.file)




# # L2 Norm
# coupling_strength = np.linalg.norm(rslds.emissions.Cs[0][:], axis=1)

# Sum of absolute loadings
# coupling_strength = np.sum(np.abs(rslds.emissions.Cs[0][:]), axis=1)

# # Mean absolute loading
# coupling_strength = np.mean(np.abs(rslds.emissions.Cs[0][:]), axis=1)

# # Maximum absolute loading
# coupling_strength = np.max(np.abs(rslds.emissions.Cs[0][:]), axis=1)

# # Sum of squared loadings
# coupling_strength = np.sum(rslds.emissions.Cs[0][:] ** 2, axis=1)

def generate_trial_figs(file, data, top_n):

    loaded = pickle.load(open(file, 'rb'))

    q_elbos, q, rslds = loaded["q_elbos"], loaded["q"], loaded["rslds"]

    pseudotrials = True if file.split('/')[-1].replace(".pkl", "").split("_")[-1] == "pseudotrials" else False
    monolith = True if file.split('/')[-1].replace(".pkl", "").split("_")[-1] == "monolith" else False
    no_fit = True if file.split('/')[-2] == "no_fit" else False

    if pseudotrials or monolith:
        mouse, activity, iterations, k, d, sb, init, ef, w, _ = file.split('/')[-1].replace(".pkl", "").split("_")
    else:
        mouse, activity, iterations, k, d, sb, init, ef, w = file.split('/')[-1].replace(".pkl", "").split("_")

    # print(coupling_strength.shape)
    # print(ranked_indices.shape)
    # print(ranked_strengths.shape)

    if data != None:
        if data.endswith(".h5"):
            with h5py.File(data, 'r') as f:
                trial_data = np.einsum('nti -> itn', f[mouse][activity][:])
        elif data.endswith(".mat"):
            raw = sio.loadmat(data)
            trial_data = np.einsum('nti -> itn', raw[f"{mouse}_{activity}"])
    else:
        if activity == 'all':
            filepath = f'data/{mouse}_activity.h5'
            with h5py.File(filepath, 'r') as f:
                trial_data = np.einsum('nti -> itn', f[mouse][activity][:])
        elif activity == 'eat3':
            filepath = f'data/{mouse}_eat3.h5'
            with h5py.File(filepath, 'r') as f:
                trial_data = np.einsum('nti -> itn', f[mouse]['eat3'][:])

            # load pellets
            raw = sio.loadmat("data/Refined_eat_drink_social_forNK.mat")
            pellets = raw[f"{mouse}_eat3_pel"][:][0]
        elif mouse in ["M1", "M2"]:
            filepath = 'data/Refined_eat_drink_social_forNK.mat'
            raw = sio.loadmat(filepath)
            trial_data = np.einsum('nti -> itn', raw[f"{mouse}_{activity}"])
        elif mouse in ["M1filtered"]:
            filepath = 'data/Refined_eat_drink_social_forNK.mat'
            raw = sio.loadmat(filepath)
            trial_data = np.einsum('nti -> itn', raw[f"M1_{activity}"])
            trial_data = trial_data[:-4]
        else:
            filepath = 'data/june1_processed_data.h5'
            with h5py.File(filepath, 'r') as f:
                trial_data = np.einsum('nti -> itn', f[mouse][activity][:])

    print(rslds.emissions.Cs.shape)

    C_matrix = rslds.emissions.Cs[0]


    all_selected_indices = {}
    all_coupling_strengths = {}

    for i in range(rslds.emissions.Cs.shape[-1]):
        # print(C_matrix[:, i].shape)
        coupling_strength = np.abs(C_matrix[:, i])
        all_coupling_strengths[f"x{i+1}"] = coupling_strength

        ranked_indices = np.argsort(coupling_strength)[::-1]
        # ranked_strengths = coupling_strength[ranked_indices]

        selected_indices = ranked_indices[:top_n]
        all_selected_indices[f"x{i+1}"] = selected_indices

    figure_root = f"figures/{mouse}_{activity}_{iterations}_{k}_{d}_{sb}_{init}_{ef}_{w}"
    if pseudotrials:
        figure_root = figure_root + "_pseudotrials"
    if monolith:
        figure_root = figure_root + "_monolith"
    if no_fit:
        figure_root = figure_root + "_nofit"

    figure_root = figure_root + f"/identify-neurons/top{top_n}"

    os.makedirs(f"{figure_root}", exist_ok=True)

    for idx, trial in enumerate(trial_data):
        plt.figure(figsize=(16, 4))
        for i in range(rslds.emissions.Cs.shape[-1]):
            selected_indices = all_selected_indices[f"x{i+1}"]
            example_trial = trial[:, selected_indices]

            ax = plt.subplot(1, 3, i + 1)
            im = ax.imshow(
            example_trial.T,
            aspect='auto',
            extent=[0, 401, example_trial.shape[1], 1],
            interpolation='none',
            cmap='magma'
        )

            ax.set_xticks([0, 100, 200, 300, 400])
            ax.set_xlabel('frames',)     # set xlabel font size
            ax.set_ylabel('neuron',)      # set ylabel font size
            ax.set_title(f'x{i+1}',)  # set title font size

            ax.tick_params(axis='both', which='major',)  # set tick label size

        plt.tight_layout() 
        plt.savefig(f"{figure_root}/trial_{idx}.png")
        plt.close("all")
    return (figure_root, all_selected_indices, rslds, all_coupling_strengths)

def plot_coupling_strengths(figure_root, all_coupling_strengths):
    for i in range(len(all_coupling_strengths)):
        plt.figure(figsize=(6, 4))
        sns.lineplot(sorted(all_coupling_strengths[f"x{i+1}"], reverse=True), color='steelblue')
        plt.xlim(0, len(all_coupling_strengths[f"x{i+1}"]))
        plt.ylim(0, max(all_coupling_strengths[f"x{i+1}"]) * 1.1)
        plt.fill_between(range(len(all_coupling_strengths[f"x{i+1}"])), sorted(all_coupling_strengths[f"x{i+1}"], reverse=True), color='steelblue')
        plt.title(f"Coupling Strengths for x{i+1}")
        plt.xlabel("Neuron Rank")
        plt.ylabel("Coupling Strength")
        plt.tight_layout()
        plt.savefig(f"{figure_root}/coupling_strengths_x{i+1}.png")
        plt.close("all")


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def generate_similarity_matrix(figure_root, selected_indices_dict_1, selected_indices_dict_2, dims, dimension_labels, output_filename):
    jaccard_similarity_matrix = np.zeros((dims, dims))
    all_selected_indices_list_1 = [set(selected_indices_dict_1[f"x{i+1}"]) for i in range(dims)]
    all_selected_indices_list_2 = [set(selected_indices_dict_2[f"x{i+1}"]) for i in range(dims)]

    for i in range(dims):
        for j in range(dims):
            jaccard_similarity_matrix[i, j] = jaccard_similarity(all_selected_indices_list_1[i], all_selected_indices_list_2[j])

    sns.heatmap(jaccard_similarity_matrix, vmin=0, vmax=1, annot=True, xticklabels=dimension_labels, yticklabels=dimension_labels)
    plt.savefig(f"{figure_root}/{output_filename}")
    plt.close("all")

    return jaccard_similarity_matrix


figure_root, all_selected_indices, rslds, all_coupling_strengths = generate_trial_figs(file=args.file, data=args.data, top_n=args.top_n)

plot_coupling_strengths(figure_root, all_coupling_strengths)

dimension_labels = [f"x{i+1}" for i in range(rslds.emissions.Cs.shape[-1])]


generate_similarity_matrix(
    figure_root=figure_root,
    selected_indices_dict_1=all_selected_indices,
    selected_indices_dict_2=all_selected_indices,
    dims=rslds.emissions.Cs.shape[-1],
    dimension_labels=dimension_labels,
    output_filename="similarity_matrix.png"
    )

# jaccard_similarity_matrix = np.zeros((rslds.emissions.Cs.shape[-1],rslds.emissions.Cs.shape[-1]))
# all_selected_indices_list = [set(all_selected_indices[f"x{i+1}"]) for i in range(rslds.emissions.Cs.shape[-1])]
# for i in range(rslds.emissions.Cs.shape[-1]):
#     for j in range(rslds.emissions.Cs.shape[-1]):
#         jaccard_similarity_matrix[i, j] = jaccard_similarity(all_selected_indices_list[i], all_selected_indices_list[j])

# dimension_labels = [f"x{i+1}" for i in range(rslds.emissions.Cs.shape[-1])]

# sns.heatmap(jaccard_similarity_matrix, vmin=0, vmax=1, annot=True, xticklabels=dimension_labels, yticklabels=dimension_labels)
# plt.savefig(f"{figure_root}/similarity_matrix.png")

if args.file2:
    figure_root_2, all_selected_indices_2, rslds_2, all_coupling_strengths_2 = generate_trial_figs(file=args.file2, data=args.data, top_n=args.top_n)

    generate_similarity_matrix(
        figure_root=figure_root_2,
        selected_indices_dict_1=all_selected_indices_2,
        selected_indices_dict_2=all_selected_indices_2,
        dims=rslds.emissions.Cs.shape[-1],
        dimension_labels=dimension_labels,
        output_filename="similarity_matrix.png"
        )

    # print(f"{args.file2.split('/')[-1].replace(".pkl", ".png")}")
    # print(f"{args.file.split('/')[-1].replace(".pkl", ".png")}")

    generate_similarity_matrix(
        figure_root=figure_root,
        selected_indices_dict_1=all_selected_indices,
        selected_indices_dict_2=all_selected_indices_2,
        dims=rslds.emissions.Cs.shape[-1],
        dimension_labels=dimension_labels,
        output_filename=f"{args.file2.split('/')[-1].replace(".pkl", "")}.png"
        )

    generate_similarity_matrix(
        figure_root=figure_root_2,
        selected_indices_dict_1=all_selected_indices_2,
        selected_indices_dict_2=all_selected_indices,
        dims=rslds.emissions.Cs.shape[-1],
        dimension_labels=dimension_labels,
        output_filename=f"{args.file.split('/')[-1].replace(".pkl", "")}.png"
        )


    
# ranked_indices = np.argsort(coupling_strength)[::-1]
# ranked_strengths = coupling_strength[ranked_indices]


# print(selected_indices)
# print(selected_indices.shape)

# print(trial_data[0][:, selected_indices].shape)

# example_trial = trial_data[0][:, selected_indices]

# fig, ax = plt.subplots()

# im = ax.imshow(
#     example_trial.T,
#     aspect='auto',
#     extent=[0, 401, example_trial.shape[1], 1],
#     interpolation='none',
#     cmap='magma'
# )

# ax.set_xticks([0, 100, 200, 300, 400])
# ax.set_xlabel('frames',)     # set xlabel font size
# ax.set_ylabel('neuron',)      # set ylabel font size
# ax.set_title('raw neural activity',)  # set title font size

# ax.tick_params(axis='both', which='major',)  # set tick label size

# fig.colorbar(im, ax=ax)

