import os
import pickle
import sys
sys.path.insert(0, "/ssm-1")
sys.modules.pop('ssm', None)
sys.modules.pop('ssm.util', None)

import copy
import h5py

import scipy.io as sio
import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(12345)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator

import pandas as pd

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import ssm


import argparse
from itertools import combinations

# Parse arguments

parser = argparse.ArgumentParser(prog="Generate Visualisations", usage="python generate_visualisations.py --file M1_eat_100.pkl")

activities = ['eat', 'drink', 'social', 'all']
parser.add_argument("-f", "--file", action="store", required=True, help="Model pickle file path", type=str)
parser.add_argument("-e", "--example-trials", action="store", required=False, default=1, help="Number of example trials to plot", type=int)
parser.add_argument("-d", "--data", action="store", required=False, default=None, help="Path of data to plot trajectory", type=str)
parser.add_argument("-a", "--activity", action="store", required=False, help="Action such as eat, drink, social", choices=activities, type=str.lower)

args = parser.parse_args()

print(f"Opening file: {args.file}")



def plot_trajectory(z, x, ax=None, ls="-", dims= [0,1], pellets=[100]):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, dims[0]],
                x[start:stop + 1, dims[1]],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                # color = "tab:gray",
                alpha = 0.8)
    ax.plot(x[-1, dims[0]], x[-1, dims[1]], 'mo', ms=5, alpha = 0.8)
    ax.plot(x[0, dims[0]], x[0, dims[1]], 'bo', ms=5, alpha=0.8)
    for pellet in pellets:
        ax.plot(x[pellet, dims[0]], x[pellet, dims[1]], 'go', ms=5, alpha=0.8)
    return ax

def plot_x1(z, x, ax=None, ls='-'):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    # for start, stop in zip(zcps[:-1], zcps[1:]):
    ax.plot(x[0:400 + 1, 0],
                lw=1, ls=ls,
                color='dimgrey',
                alpha=1.0)
    ax.axvline(x=0, color='b', linestyle='--', lw=1)
    ax.axvline(x=100, color='g', linestyle='--', lw=1)
    ax.axvline(x=400, color='m', linestyle='--', lw=1)

    ax.set_xlabel("timesteps")
    ax.set_ylabel("$x_1$")
    return ax

def plot_x2(z, x, ax=None, ls='-'):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    # for start, stop in zip(zcps[:-1], zcps[1:]):
    ax.plot(x[0:400 + 1, 1],
                lw=1, ls=ls,
                color='dimgrey',
                alpha=1.0)
    ax.axvline(x=0, color='b', linestyle='--')
    ax.axvline(x=100, color='g', linestyle='--')
    ax.axvline(x=400, color='m', linestyle='--')

    ax.set_xlabel("timesteps")
    ax.set_ylabel("$x_2$")
    return ax

def plot_dim(z, x, dim, events_df, ax=None, ls='-', pellets=[100]):
    # zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    # print(x.shape[0])
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    # for start, stop in zip(zcps[:-1], zcps[1:]):
    ax.plot(x[0:x.shape[0] + 1, dim],
                lw=1, ls=ls,
                color='dimgrey',
                alpha=1.0)
    ax.axvline(x=0, color='b', linestyle='--')
    ax.axvline(x=x.shape[0], color='m', linestyle='--')
    for pellet in pellets:
        ax.axvline(x=pellet, color='g', linestyle='--', alpha=0.5, lw=0.5)
    for i, row in events_df.iterrows():
        if row["show"]:
            ax.axvline(x=row["frame"], color='tab:gray', linestyle='--', lw=0.5)
        ax.text(x=row["frame"], y=-1.5, s=row["Type"], rotation=90, fontsize=8, alpha=0.5)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"$x_{dim+1}$")
    return ax

def plot_observations(z, y, ax=None, ls="-", lw=1):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start: stop + 1], y[start:stop + 1, n],
                    w=lw, ls=ls,
                    color=colors[z[start] % len(colors)],
                    alpha=1.0)
    return ax

def plot_most_likely_dynamics(model,
                              xlim=(-4,4), ylim=(-3,3), nxpts=20, nypts=20,
                              alpha=0.8, ax=None, figsize=(3, 3), dims=[0,1]):
    
    K = model.K
    # assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the most likely state at each xy location
    logits = xy.dot(model.transitions.Rs.T[np.ix_(dims)]) + model.transitions.r
    z = np.argmax(logits, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A[np.ix_(dims, dims)].T) + b[np.ix_(dims)] - xy

        zk = z == k

        if np.any(zk):
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha
                    )
            
    ax.set_xlabel(f'$x_{dims[0] + 1}$')
    ax.set_ylabel(f'$x_{dims[1] + 1}$')

    plt.tight_layout()

    return ax

def get_all_event_labels(mouse):
    # this will return all the event labels for the given mouse
    filepath = f"./data/raw/{mouse}_events.csv"
    df = pd.read_csv(filepath)
    df = df.drop(['Date_Time'], axis=1)
    return df

def get_trial_event_labels(events_df, index, activity, steps_before=100, steps_after=300, absolute_start_frame=False):

    activity_conversion_dict = {
        "eat" : ["eat_start", "retrieve_pellet"],
        "drink": ["drink_start", "enter_drink"],
        "social": ["social_long_start", "social_short_start", "enter_social"],
        "eat3": ["eat3_start"]
    }
    activity_event = activity_conversion_dict[activity]
    matched_event = events_df[events_df['Type'].isin(activity_event)].iloc[index]
    matched_event_frame = matched_event["frame"]

    start_frame = matched_event_frame - steps_before
    end_frame = matched_event_frame + steps_after
    filtered_events = events_df[events_df["frame"].between(start_frame, end_frame)]
    if not absolute_start_frame:
        filtered_events["frame"] = filtered_events["frame"] - start_frame

    filtered_events["show"] = ~filtered_events["Type"].isin(activity_event)
    filtered_events = filtered_events[filtered_events["Type"] != "eat3_start"]

    if activity == 'eat3':
        filtered_events["show"] = ~filtered_events["Type"].isin(activity_conversion_dict["eat"])

    return filtered_events

def get_mouse_activity(mouse, activity, rslds, events_df, figure_root):
    activity_conversion_dict = {
            "eat" : ["eat_start", "retrieve_pellet"],
            "drink": ["drink_start", "enter_drink"],
            "social": ["social_long_start", "social_short_start", "enter_social"],
            "eat3": ["eat3_start"]
        }
    activity_event = activity_conversion_dict[activity]

    events_df = copy.deepcopy(events_df)

    filepath = f"./data/raw/{mouse}_activity.csv"
    df = pd.read_csv(filepath, header=None)
    print(df.to_numpy().T.shape)
    activity_data = df.to_numpy().T
    q_elbos, q = rslds.approximate_posterior(activity_data, method="laplace_em", num_iters=1)
    # print(len(q.mean_continuous_states))
    # print(q.mean_continuous_states[0].shape)
    xhat = q.mean_continuous_states[0]
    events_df = events_df[events_df["Type"] != "eat3_start"]
    print(activity_event)
    if activity == "eat3":
        activity_event = activity_event + activity_conversion_dict["eat"]
    # print(activity_event)
    events_df["show"] = ~events_df["Type"].isin(activity_event)
    pellets = events_df[events_df["Type"].isin(activity_event)]["frame"].tolist()

    # print(events_df)

    # print(pellets)
    n_dims = int(rslds.dynamics.As[0].shape[0])

    plt.figure(figsize=(100, 4 * n_dims))

    for i in range(n_dims):
        ax = plt.subplot(n_dims, 1, i + 1)
        plot_dim(z=None, x=xhat, dim=i, events_df=events_df, ax=ax, pellets=pellets)
        ax.set_xlim(left=0)
    plt.tight_layout()
    # ax = plt.subplot(3, 1, 1)
    # ax2 = plt.subplot(3, 1, 2)
    # ax3 = plt.subplot(3, 1, 3)
    # plot_dim(z=None, x=xhat, dim=0, events_df=events_df, ax=ax, pellets=pellets)
    # ax.set_xlim(left=0)
    # plot_dim(z=None, x=xhat, dim=1, events_df=events_df, ax=ax2, pellets=pellets)
    # ax2.set_xlim(left=0)
    # plot_dim(z=None, x=xhat, dim=2, events_df=events_df, ax=ax3, pellets=pellets)
    # ax3.set_xlim(left=0)

    # plt.tight_layout()
    plt.savefig(f"{figure_root}/flowfield/complete_activity.png")
    plt.close("all")



    
def plot_all_trials_dim(x, dim, events_df, ax=None, ls='-', pellets=[100]):
    # zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    # print(x.shape[0])
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    # for start, stop in zip(zcps[:-1], zcps[1:]):
    ax.plot(x[0:x.shape[0] + 1, dim],
                lw=1, ls=ls,
                color='dimgrey',
                alpha=1.0)
    for i in range(len(events_df) % 401):
        ax.axvline(x=i*401, color='k', linestyle='--', alpha=0.5, lw=0.5)
        ax.text(x=i*401, y=-1.5, s=f"Trial {i}", rotation=90, fontsize=8, alpha=1)
        for j, row in events_df[i].iterrows():
            if row["show"]:
                ax.axvline(x=row["frame"] + i*401, color='tab:gray', linestyle='--', lw=0.5)
            else:
                ax.axvline(x=row["frame"] + i*401, color='g', linestyle='--', alpha=0.5, lw=0.5)
            ax.text(x=row["frame"] + i*401, y=-1.5, s=row["Type"], rotation=90, fontsize=8, alpha=0.5)

    # ax.axvline(x=x.shape[0], color='m', linestyle='--')
    # for pellet in pellets:
    #     ax.axvline(x=pellet, color='g', linestyle='--', alpha=0.5, lw=0.5)
    # ax.set_xlabel("timesteps")
    # ax.set_ylabel(f"$x_{dim+1}$")
    return ax

pseudotrials = True if args.file.split('/')[-1].replace(".pkl", "").split("_")[-1] == "pseudotrials" else False
monolith = True if args.file.split('/')[-1].replace(".pkl", "").split("_")[-1] == "monolith" else False
no_fit = True if args.file.split('/')[-2] == "no_fit" else False

if pseudotrials or monolith:
    mouse, activity, iterations, k, d, sb, init, ef, w, _ = args.file.split('/')[-1].replace(".pkl", "").split("_")
else:
    mouse, activity, iterations, k, d, sb, init, ef, w = args.file.split('/')[-1].replace(".pkl", "").split("_")

if args.data != None:
    if args.data.endswith(".h5"):
        with h5py.File(args.data, 'r') as f:
            trial_data = np.einsum('nti -> itn', f[mouse][args.activity][:])
    elif args.data.endswith(".mat"):
        raw = sio.loadmat(args.data)
        trial_data = np.einsum('nti -> itn', raw[f"{mouse}_{args.activity}"])
else:
    if activity == 'all':
        filepath = f'data/{mouse}_activity.h5'
        with h5py.File(filepath, 'r') as f:
            trial_data = np.einsum('nti -> itn', f[mouse][activity][:])
    elif activity == 'eat3long' or activity == 'eat3':
        filepath = f'data/{mouse}_{activity}.h5'
        with h5py.File(filepath, 'r') as f:
            trial_data = np.einsum('nti -> itn', f[mouse][activity][:])

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


datas = [trial_data[i] for i in range(trial_data.shape[0])]

loaded = pickle.load(open(args.file, 'rb'))

q_elbos, q, rslds = loaded["q_elbos"], loaded["q"], loaded["rslds"]


figure_root = f"figures/{mouse}_{activity}_{iterations}_{k}_{d}_{sb}_{init}_{ef}_{w}"
if pseudotrials:
    figure_root = figure_root + "_pseudotrials"
if monolith:
    figure_root = figure_root + "_monolith"
if no_fit:
    figure_root = figure_root + "_nofit"


os.makedirs(f"{figure_root}/flowfield/onlyflow/", exist_ok=True)
os.makedirs(f"{figure_root}/x1_x2", exist_ok=True)

# Eigenspectra Analysis
def calculate_time_constant(eigenvalues):
    res = [np.abs(0.1/np.log(eigenvalue)) for eigenvalue in eigenvalues]
    return res

fig, axes = plt.subplots(len(rslds.dynamics.As), 2, figsize=(12, 4*len(rslds.dynamics.As)))
for idx, matrix in enumerate(rslds.dynamics.As):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # print(f"eigenvalues: {eigenvalues}")
    # print(f"eigenvectors: {eigenvectors}")

    time_constants = calculate_time_constant(eigenvalues)
    dimensions = []
    for i in range(int(list(d)[1])):
        dimensions.append(f"$x_{i+1}$")
    time_constants_df = pd.DataFrame({"Dimension": dimensions,
                                   "Time constants / s":time_constants})
    line_attractor_score = np.log2(np.partition(time_constants, -1)[-1]/np.partition(time_constants, -2)[-2])
    # print(f"Time constrants: {time_constants}")
    # print(f"Line attractor score: {line_attractor_score}")
    sns.barplot(time_constants_df, x="Dimension", y="Time constants / s", ax=axes[idx, 0], color=colors[idx]).set_title("Time Constants")
    axes[idx, 1].text(0.2, 0.5, f"Line attractor score: {line_attractor_score.round(decimals=3)}")
    axes[idx, 1].axis("off")
# print(rslds.dynamics.As.shape)
plt.tight_layout()
plt.savefig(f"{figure_root}/eigenspectra.png")
plt.close("all")



sns.lineplot(q_elbos[1:])

plt.tight_layout()
plt.savefig(f"{figure_root}/elbos.png")
plt.close("all")


xhat_mean = np.max(np.abs(q.mean_continuous_states), axis=1)

rslds = copy.deepcopy(rslds)


# Plot example trajectories
n_trials_to_plot = min(args.example_trials, len(datas)) if args.example_trials != -1 else len(datas)
example_trial_indices = np.random.choice(range(len(datas)), n_trials_to_plot, replace=False)

# print(pellets[0][0][0])

all_events_df = get_all_event_labels(mouse=mouse)

get_mouse_activity(mouse=mouse, activity=activity, rslds=rslds, events_df=all_events_df, figure_root=figure_root)

for example_trial_index in example_trial_indices:
    # print(example_trial_index)

    trial_events_df = get_trial_event_labels(events_df=all_events_df, index=example_trial_index, activity=activity)
    # print(trial_events_df)

    # print(f"hello {example_trial_index}")

    xhat = q.mean_continuous_states[example_trial_index]
    zhat = rslds.most_likely_states(xhat, datas[example_trial_index])
    
    n_dims = xhat.shape[1]
    lim = abs(xhat_mean).max(axis=0) + 1
    # if n_dims == 2:
    #     plt.figure(figsize=(18,4))
    #     ax = plt.subplot(131)
    #     plot_most_likely_dynamics(rslds, xlim=(-lim[0],lim[0]), ylim=(-lim[1],lim[1]), ax=ax, dims=[0,1])
        
    #     plot_trajectory(zhat, xhat, ax=ax, dims=[0,1])

    #     ax2 = plt.subplot(132)
    #     ax3 = plt.subplot(133)
    #     # plot_x1(zhat, xhat, ax2)
    #     # plot_x2(zhat, xhat, ax3)
    #     plot_dim(zhat, xhat, 0, ax2)
    #     plot_dim(zhat, xhat, 1, ax3)

    #     plt.title("Inferred Dynamics")
    #     plt.tight_layout()
    #     plt.savefig(f"{figure_root}/flowfield/trial_{example_trial_index}.png")


    #     plt.close("all")
    # elif n_dims == 3:
    dim_combinations = list(combinations(range(n_dims), 2))
    n_rows = len(dim_combinations)
    plt.figure(figsize=(18, 4 * n_rows))
    for idx, val in enumerate(dim_combinations):
        # ax = plt.subplot((31 + n_rows * 100) * subplot_modifier + idx * 3)
        ax = plt.subplot(n_rows, 3, 1 + idx * 3)
        plot_most_likely_dynamics(rslds, xlim=(-lim[0],lim[0]), ylim=(-lim[1],lim[1]), ax=ax, dims=val)

        if activity == "eat3" or activity == 'eat3long':
            pels = pellets[example_trial_index][0][0]
            pellets_to_plot = []
            offset = 0 if activity == 'eat3' else 900
            for pel in pels:
                pellets_to_plot.append(int(pel) + offset)

            plot_trajectory(zhat, xhat, ax=ax, dims=val, pellets=pellets_to_plot)
            
            ax2 = plt.subplot(n_rows, 3, 2 + idx * 3)
            # print(n_rows, idx, n_dims, 3 + idx * 3)
            ax3 = plt.subplot(n_rows, 3, 3 + idx * 3)
            plot_dim(zhat, xhat, val[0], trial_events_df, ax2, pellets=pellets_to_plot)
            plot_dim(zhat, xhat, val[1], trial_events_df, ax3, pellets=pellets_to_plot)
        else:
            plot_trajectory(zhat, xhat, ax=ax, dims=val)

            ax2 = plt.subplot(n_rows, 3, 2 + idx * 3)
            ax3 = plt.subplot(n_rows, 3, 3 + idx * 3)
            plot_dim(zhat, xhat, val[0], trial_events_df, ax2)
            plot_dim(zhat, xhat, val[1], trial_events_df, ax3)
    # for idx in range(n_dims):
    #     ax = plt.subplot(234 + idx)
    #     plot_dim(zhat, xhat, idx, ax)

    plt.tight_layout()
    plt.title("Inferred Dynamics")
    plt.savefig(f"{figure_root}/flowfield/trial_{example_trial_index}.png")
    axes = plt.gcf().get_axes()
    for i in range(len(dim_combinations)):
        for j in [1,2]:
            # print(i * 3 + j)
            axes[i * 3 + j].set_visible(False)

    plt.savefig(f"{figure_root}/flowfield/onlyflow/trial_{example_trial_index}.png", bbox_inches='tight')
    plt.close("all")

plt.figure(figsize=(18, 4 * int(d[1:])))
data_flat = trial_data.reshape(-1, trial_data.shape[-1])
all_trials_events = []
for idx, data in enumerate(datas):
    all_trials_events.append(get_trial_event_labels(all_events_df, idx, activity))

xhat = np.array(q.mean_continuous_states)
# print(xhat.shape)
xhat_flat = xhat.reshape(-1, xhat.shape[-1])
# print(xhat_flat.shape)

for i in range(int(d[1:])):
    ax = plt.subplot(int(d[1:]), 1, i + 1)
    ax.xaxis.set_visible(False)
    plot_all_trials_dim(xhat_flat, i, all_trials_events, ax)

plt.tight_layout()
# plt.title()
plt.savefig(f"{figure_root}/flowfield/all_trials_consecutive.png")
plt.close("all")

# Variance across trials
# print(xhat.shape)
# print(np.var(xhat, axis=1).shape)
# print(np.var(xhat, axis=1))
var = np.var(xhat, axis=1)
fig, axes = plt.subplots(var.shape[-1], 1, sharex=True,
                         figsize=(18, 4 * var.shape[-1]))
axes = np.atleast_1d(axes)
ticks = np.arange(var.shape[0])
for i in range(var.shape[-1]):
    ax = axes[i]
    sns.lineplot(var[:, i], ax=ax)
    ax.set_ylabel(f"Variance of x{i + 1}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
plt.xlim(left=0)
axes[-1].set_xlabel("Trial index")
plt.tight_layout()
plt.savefig(f"{figure_root}/flowfield/variance_across_trials.png")

# for i in range(xhat_flat.shape[1]):
#     var = np.var(xhat, axis=0)[i]
#     print(f"Variance of dimension {i}: {var}")
#     sns.lineplot(var)
#     plt.show()
#     plt.close("all")

# print(data_flat.shape)

    # plt.savefig(f"{figure_root}/x1_x2/trial_{example_trial_index}.png")
    # plt.close("all")
    




