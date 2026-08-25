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
from sklearn.model_selection import KFold

import ssm

import argparse

from statsmodels.tsa.api import SimpleExpSmoothing

from affinewarp import PiecewiseWarping
import pandas as pd

# Parse Arguments

parser = argparse.ArgumentParser(prog="Train rSLDS", usage="python train_rslds.py --mouse M1 --activity eat --iterations 1000")

mice = ["M1", "M2", "g04", "g05", "g06", "g12", "g14", "g19", "M1filtered"]
activities = ['eat', 'drink', 'social', 'all', 'eat3long', 'eat3']
parser.add_argument("-m", "--mouse", action="store", required=True, help="Name of mouse", type=str, choices=mice)
parser.add_argument("-a", "--activity", action="store", required=True, help="Action such as eat, drink, social", choices=activities, type=str.lower)
parser.add_argument("-i", "--iterations", action="store", required=False, default=100, help="Number of iteration to train rSLDS", type=int)
parser.add_argument("-f", "--file", action="store", required=False, default=None, help="Data file path", type=str)
parser.add_argument("-k", "--k-regions", action="store", required=False, default=4, help="Number of piecewise regions to learn for dynamics", type=int)
parser.add_argument("-d", "--latent-dimensions", action="store", required=False, default=2, help="Number of latent dimensions", type=int)
parser.add_argument("-sb", "--subtract-baseline", action="store_true", required=False, default=False, help="Subtract baseline F0")
parser.add_argument("-in", "--initialization", action="store", required=False, default="PCA", help="Initialisation of the model")
parser.add_argument("-nf", "--no-fit", action="store_true", required=False, default=False, help="Do not fit model")
parser.add_argument("-p", "--pseudotrials", action="store_true", required=False, default=False, help="Train on pseudotrials")
parser.add_argument("-nrm", "--normalised", action="store_true", required=False, default=False, help="Normalise activity")
parser.add_argument("-ef", "--exponential-filter", action="store", required=False, default=-1, help="Exponential filter", type=float)
parser.add_argument("-mon", "--monolith", action="store_true", required=False, default=False, help="Train on all data")
parser.add_argument("-w", "--warp", action="store_true", required=False, default=False, help="Piecewise Warp data")
parser.add_argument("-cv", "--cross-validate", action="store_true", required=False, default=False, help="Cross Validation")

args = parser.parse_args()

assert args.mouse in mice and isinstance(args.mouse, str)
assert args.activity in activities and isinstance(args.activity, str)
assert isinstance(args.iterations, int)

eval_data = None

if args.pseudotrials == True or args.monolith == True:
    if args.activity == "eat3":
        filepath = f'data/{args.mouse}_eat3.h5' if args.file == None else args.file
        with h5py.File(filepath, 'r') as f:
            eval_data = np.einsum('nti -> itn', f[args.mouse]['eat3'][:])
    elif args.activity == "eat3long":
        filepath = f'data/{args.mouse}_eat3long.h5' if args.file == None else args.file
        with h5py.File(filepath, 'r') as f:
            eval_data = np.einsum('nti -> itn', f[args.mouse]['eat3long'][:])
    elif args.mouse in ["M1", "M2"]:
        # print("old mouse found")
        filepath = 'data/Refined_eat_drink_social_forNK.mat' if args.file == None else args.file
        raw = sio.loadmat(filepath)
        eval_data = np.einsum('nti -> itn', raw[f"{args.mouse}_{args.activity}"])
    else:
        filepath = 'data/june1_processed_data.h5' if args.file == None else args.file
        with h5py.File(filepath, 'r') as f:
            eval_data = np.einsum('nti -> itn', f[args.mouse][args.activity][:])

# Load Data
if args.monolith == True:
    filepath = f'data/{args.mouse}_activity_monolith.h5'
    with h5py.File(filepath, 'r') as f:
        trial_data = np.einsum('nti -> itn', f[args.mouse]["monolith"][:])
elif args.activity == 'all' or args.pseudotrials == True:
    filepath = f'data/{args.mouse}_activity.h5'
    with h5py.File(filepath, 'r') as f:
        trial_data = np.einsum('nti -> itn', f[args.mouse]["all"][:])
        # if args.mouse == 'M1filtered':
        #     trial_data = trial_data[:-3000]
elif args.activity == 'eat3long':
    filepath = f'data/{args.mouse}_eat3long.h5'
    with h5py.File(filepath, 'r') as f:
        trial_data = np.einsum('nti -> itn', f[args.mouse]['eat3long'][:])
elif args.activity == 'eat3':
    filepath = f'data/{args.mouse}_eat3.h5'
    with h5py.File(filepath, 'r') as f:
        eval_data = np.einsum('nti -> itn', f[args.mouse]['eat3'][:])
    filepath = 'data/Refined_eat_drink_social_forNK.mat' if args.file == None else args.file
    raw = sio.loadmat(filepath)
    trial_data = np.einsum('nti -> itn', raw[f"{args.mouse}_eat"])
elif args.mouse in ["M1", "M2"]:
    # print("old mouse found")
    filepath = 'data/Refined_eat_drink_social_forNK.mat' if args.file == None else args.file
    raw = sio.loadmat(filepath)
    trial_data = np.einsum('nti -> itn', raw[f"{args.mouse}_{args.activity}"])
elif args.mouse in ["M1filtered"]:
    filepath = 'data/Refined_eat_drink_social_forNK.mat' if args.file == None else args.file
    raw = sio.loadmat(filepath)
    trial_data = np.einsum('nti -> itn', raw[f"M1_{args.activity}"])
    trial_data = trial_data[:-4]
else:
    # print("new mouse found")
    filepath = 'data/june1_processed_data.h5' if args.file == None else args.file
    with h5py.File(filepath, 'r') as f:
        trial_data = np.einsum('nti -> itn', f[args.mouse][args.activity][:])

def subtract_baseline(data, Fs=10):
    if not args.subtract_baseline:
        return (data, 0)
    baseline = findBaselineF0(data, Fs, axis=1, keepdims=True) 
    baseline_subtracted_data = data - baseline
    return (baseline_subtracted_data, baseline)

def normalise_data(data):
    if not args.normalised:
        return (data, 0, 1)

    data = (data - data.mean(0)) / data.std(0)
    return (data, data.mean(0), data.std(0))

def train_rslds(data,
                slds_T=1000,
                slds_K=args.k_regions,
                slds_D_latent=args.latent_dimensions,
                slds_transitions="recurrent_only",
                slds_dynamics="diagonal_gaussian",
                slds_emissions="gaussian_orthog",
                slds_single_subspace=True,
                initialization_method = args.initialization,
                fit_method="laplace_em",
                fit_variational_posterior="structured_meanfield",
                fit_initialize=False,
                fit_num_iters=100,
                fit_alpha=0.0,
                data_baseline=0,
                data_mean=0,
                data_std=1,
                data_eval_data=[]):

    if args.exponential_filter > 0:

        smoothing_level = 1 / args.exponential_filter

        smoothed_data = []
        for trial in data:

            smooth_trial = []
            for neuron in trial.T:
                fit = SimpleExpSmoothing(neuron, initialization_method="heuristic").fit(smoothing_level=smoothing_level, optimized=False)
                smooth_trial.append(fit.fittedvalues)
            smooth_trial = np.array(smooth_trial)
            smoothed_data.append(smooth_trial.T)
        data = np.array(smoothed_data)

    if args.warp:
        warp_model = PiecewiseWarping(n_knots=0, warp_reg_scale=1e-6, smoothness_reg_scale=20.0)

        warp_model.fit(data=data, iterations=50, warp_iterations=200)

        data = warp_model.transform(data)

    datas = [data[i] for i in range(data.shape[0])]

    rslds = ssm.SLDS(data.shape[2], slds_K, slds_D_latent,
                        transitions=slds_transitions,
                        dynamics=slds_dynamics,
                        emissions=slds_emissions,
                        single_subspace=slds_single_subspace)
    if initialization_method == "PCA":
        rslds.initialize(datas)
    elif initialization_method == "FA":
        fa = FactorAnalysis(n_components=slds_D_latent)
        X_concat = np.concatenate(datas, axis=0)
        X_init = fa.fit_transform(X_concat)
        
        # FactorAnalysis.components_ has shape (n_components, n_features)
        # We need emissions.Cs to be (1, N, D) when single_subspace=True
        comps = fa.components_.T  # shape (N, D)
        # Orthogonalize components to satisfy GaussianOrthog emissions constraint
        U, S, Vt = np.linalg.svd(comps, full_matrices=False)
        Cs_orth = U[:, :slds_D_latent]  # shape (N, D)
        rslds.emissions.Cs = Cs_orth.reshape((1, Cs_orth.shape[0], Cs_orth.shape[1]))
        rslds.emissions.d = fa.mean_

    if args.no_fit:
        q_elbos, q = rslds.approximate_posterior(datas, method=fit_method, num_iters=1)
        # q_elbos = np.zeros((fit_num_iters))
    else:
        q_elbos, q = rslds.fit(datas,
                        method=fit_method,
                        variational_posterior=fit_variational_posterior,
                        initialize=fit_initialize,
                        num_iters=fit_num_iters,
                        alpha=fit_alpha)

    if args.pseudotrials == True or args.monolith == True or data_eval_data is not None:
        # print(eval_data)
        # eval_data = eval_data - data_baseline
        # eval_data = eval_data - data_mean
        # eval_data = eval_data / data_std
        data_eval_data = (data_eval_data - data_baseline.mean(0) - data_mean) / data_std
        eval_datas = [data_eval_data[i] for i in range(data_eval_data.shape[0])]
        _, q = rslds.approximate_posterior(eval_datas, method=fit_method, num_iters=1)
    
    to_pickle = {"q_elbos": q_elbos, "q": q, "rslds": rslds}

    save_file_name = f'{args.mouse}_{args.activity}_{args.iterations}_k{args.k_regions}_d{args.latent_dimensions}_sb{args.subtract_baseline}_{args.initialization}_ef{args.exponential_filter}_w{args.warp}'
    if args.pseudotrials == True:
        save_file_name = save_file_name + "_pseudotrials"

    if args.monolith == True:
        save_file_name = save_file_name + "_monolith"
    if args.no_fit:
        os.makedirs(f"./pickle/no_fit", exist_ok=True)
        pickle.dump(to_pickle, open(f'pickle/no_fit/{save_file_name}.pkl', 'wb'))
    else:
        os.makedirs(f"./pickle/", exist_ok=True)
        pickle.dump(to_pickle, open(f'./pickle/{save_file_name}.pkl', 'wb'))

    return None

def test_rslds(data,
                slds_T=1000,
                slds_K=args.k_regions,
                slds_D_latent=args.latent_dimensions,
                slds_transitions="recurrent_only",
                slds_dynamics="diagonal_gaussian",
                slds_emissions="gaussian_orthog",
                slds_single_subspace=True,
                initialization_method = args.initialization,
                fit_method="laplace_em",
                fit_variational_posterior="structured_meanfield",
                fit_initialize=False,
                fit_num_iters=100,
                fit_alpha=0.0,
                data_baseline=0,
                data_mean=0,
                data_std=1,
                data_eval_data=[]):
        
    if args.exponential_filter > 0:
    
        smoothing_level = 1 / args.exponential_filter

        smoothed_data = []
        for trial in data:

            smooth_trial = []
            for neuron in trial.T:
                fit = SimpleExpSmoothing(neuron, initialization_method="heuristic").fit(smoothing_level=smoothing_level, optimized=False)
                smooth_trial.append(fit.fittedvalues)
            smooth_trial = np.array(smooth_trial)
            smoothed_data.append(smooth_trial.T)
        data = np.array(smoothed_data)
    if args.warp:
        warp_model = PiecewiseWarping(n_knots=0, warp_reg_scale=1e-6, smoothness_reg_scale=20.0)

        warp_model.fit(data=data, iterations=50, warp_iterations=200)

        data = warp_model.transform(data)

    datas = [data[i] for i in range(data.shape[0])]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    forecast_start = 110   # python index
    forecast_end = 400     # exclusive

    def r2_score_np(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
        if ss_tot <= 1e-12:
            return np.nan
        return 1.0 - (ss_res / ss_tot)

    def latent_rollout_from_prefix(rslds_model, y_seq, start_t, end_t, post_iters=25):
        """
        Infer latent up to start_t, then open-loop rollout in latent space
        using continuous dynamics with last inferred discrete state.
        """
        if y_seq.shape[0] < end_t or start_t < 1:
            return None

        _, q_prefix = rslds_model.approximate_posterior([y_seq[:start_t]], method=fit_method, num_iters=post_iters)
        x_prefix = q_prefix.mean_continuous_states[0]  # shape (start_t, D)
        z_prefix = rslds_model.most_likely_states(x_prefix, y_seq[:start_t])

        x_prev = x_prefix[-1].copy()
        z_prev = int(z_prefix[-1])

        As = rslds_model.dynamics.As  # (K, D, D)
        bs = rslds_model.dynamics.bs  # (K, D)

        H = end_t - start_t
        x_pred = np.zeros((H, x_prev.shape[0]))

        # Simple open-loop: keep last inferred discrete mode fixed
        for h in range(H):
            x_prev = As[z_prev].dot(x_prev) + bs[z_prev]
            x_pred[h] = x_prev

        return x_pred

    for i, (train_index, test_index) in enumerate(kf.split(datas)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        rslds = ssm.SLDS(data.shape[2], slds_K, slds_D_latent,
                                transitions=slds_transitions,
                                dynamics=slds_dynamics,
                                emissions=slds_emissions,
                                single_subspace=slds_single_subspace)
        
        train_data = [datas[i] for i in train_index]
        test_data = [datas[i] for i in test_index]

        if initialization_method == "PCA":
            rslds.initialize(train_data)
        elif initialization_method == "FA":
            fa = FactorAnalysis(n_components=slds_D_latent)
            X_concat = np.concatenate(train_data, axis=0)
            X_init = fa.fit_transform(X_concat)
            
            # FactorAnalysis.components_ has shape (n_components, n_features)
            # We need emissions.Cs to be (1, N, D) when single_subspace=True
            comps = fa.components_.T  # shape (N, D)
            # Orthogonalize components to satisfy GaussianOrthog emissions constraint
            U, S, Vt = np.linalg.svd(comps, full_matrices=False)
            Cs_orth = U[:, :slds_D_latent]  # shape (N, D)
            rslds.emissions.Cs = Cs_orth.reshape((1, Cs_orth.shape[0], Cs_orth.shape[1]))
            rslds.emissions.d = fa.mean_

        # Train ELBO
        train_elbos, q_train_full = rslds.fit(
            train_data,
            method=fit_method,
            variational_posterior=fit_variational_posterior,
            initialize=fit_initialize,
            num_iters=fit_num_iters,
            alpha=fit_alpha
        )

        # Test ELBO (held-out predictive proxy)
        test_elbos, q_test_full = rslds.approximate_posterior(
            test_data, method=fit_method, num_iters=25
        )

        train_elbo = float(train_elbos[-1])
        test_elbo = float(test_elbos[-1])

        train_elbo_per_obs = train_elbo / len(train_data)
        test_elbo_per_obs = test_elbo / len(test_data)

        x_train_true = q_train_full.mean_continuous_states
        x_test_true = q_test_full.mean_continuous_states

        # Forecast and collect [start:end) windows
        train_true_chunks, train_pred_chunks = [], []
        for y_seq, x_true in zip(train_data, x_train_true):
            x_pred = latent_rollout_from_prefix(rslds, y_seq, forecast_start, forecast_end, post_iters=25)
            if x_pred is None:
                continue
            train_true_chunks.append(x_true[forecast_start:forecast_end])
            train_pred_chunks.append(x_pred)

        test_true_chunks, test_pred_chunks = [], []
        for y_seq, x_true in zip(test_data, x_test_true):
            x_pred = latent_rollout_from_prefix(rslds, y_seq, forecast_start, forecast_end, post_iters=25)
            if x_pred is None:
                continue
            test_true_chunks.append(x_true[forecast_start:forecast_end])
            test_pred_chunks.append(x_pred)

        train_r2 = np.nan
        test_r2 = np.nan
        if len(train_true_chunks) > 0:
            train_r2 = r2_score_np(np.concatenate(train_true_chunks, axis=0),
                                   np.concatenate(train_pred_chunks, axis=0))
        if len(test_true_chunks) > 0:
            test_r2 = r2_score_np(np.concatenate(test_true_chunks, axis=0),
                                  np.concatenate(test_pred_chunks, axis=0))

        print(f"Fold {i}:")
        print(f"  train ELBO total     = {train_elbo:.3f}")
        print(f"  test  ELBO total     = {test_elbo:.3f}")
        print(f"  train ELBO / obs     = {train_elbo_per_obs:.6f}")
        print(f"  test  ELBO / obs     = {test_elbo_per_obs:.6f}")
        print(f"  train latent R^2     = {train_r2:.6f}")
        print(f"  test  latent R^2     = {test_r2:.6f}")

        fold_metrics.append({
            "fold": i,
            "train_elbo": train_elbo,
            "test_elbo": test_elbo,
            "train_elbo_per_obs": train_elbo_per_obs,
            "test_elbo_per_obs": test_elbo_per_obs,
            "train_latent_r2_100_400": float(train_r2) if not np.isnan(train_r2) else np.nan,
            "test_latent_r2_100_400": float(test_r2) if not np.isnan(test_r2) else np.nan,
        })

    save_file_name = f'{args.mouse}_{args.activity}_{args.iterations}_k{args.k_regions}_d{args.latent_dimensions}_sb{args.subtract_baseline}_{args.initialization}_ef{args.exponential_filter}_w{args.warp}'
    if args.pseudotrials == True:
        save_file_name = save_file_name + "_pseudotrials"

    if args.monolith == True:
        save_file_name = save_file_name + "_monolith"

    metrics_df = pd.DataFrame(fold_metrics)
    os.makedirs(f"./results/", exist_ok=True)
    metrics_df.to_csv(f"./results/{save_file_name}.csv")

    return fold_metrics

baseline_subtracted_data, baseline = subtract_baseline(trial_data, Fs=10)
baseline_subtracted_normalised_data, mean, std = normalise_data(baseline_subtracted_data)

if args.cross_validate:
   test_rslds(data = baseline_subtracted_normalised_data, fit_num_iters=args.iterations, data_baseline=baseline, data_mean = mean, data_std = std, data_eval_data=eval_data)
else:
    train_rslds(data = baseline_subtracted_normalised_data, fit_num_iters=args.iterations, data_baseline=baseline, data_mean = mean, data_std = std, data_eval_data=eval_data)