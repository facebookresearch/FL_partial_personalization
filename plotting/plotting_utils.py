
import math
import numpy as np
import pandas as pd
import pickle as pkl
import os, sys, time, copy
import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['axes.titlesize'] = 18

#########################################
########### Global variables
#######################################
pfl_algo_rename_dict = dict(
    ft = "Finetune",
    finetune = "Finetune",
    pfl_fl = "Finetune",
    pfl_am = "PFL-AM",
    pfl_su = "PFL-SU",
    pfl_alternating = "PFL-AM",
    pfl_joint = "PFL-SU"
)

dataset_rename_dict = dict(
    emnist_resnetgn= "EMNIST (ResNet)",
    gldv2_resnetgn="GLDv2 (ResNet)",
    so_mini="StackOverflow (Transformer)",
    so_tiny="StackOverflow (Transformer tiny)"
)

metric_rename_dict = {
    "accuracy|mean": "Mean Accuracy \%",
    "accuracy|std": "Std. of Accuracy \%",
    "accuracy|quantile_0.1": "10th Percentile Accuracy \%",
    "accuracy|quantile_0.25": "25th Percentile Accuracy \%",
    "accuracy|quantile_0.5": "Median Accuracy \%"
}

from socket import gethostname
if 'devfair' in gethostname():
    MAIN_DIR = '/checkpoint/pillutla/pfl'
elif 'pillutla-mpb' in gethostname():
    MAIN_DIR = '/Users/pillutla/Dropbox (Facebook)/code/pfl_outputs'
elif 'zh-' in gethostname():
    MAIN_DIR = '/home/pillutla/fl/pfl_outputs'
else:  # personal laptop
    MAIN_DIR = None

OUTPUT_DIR_FL = f"{MAIN_DIR}/outputs2"
OUTPUT_DIR_PFL = f"{MAIN_DIR}/outputs3"


#########################################
########### Data loading utils
#######################################
def load_pkl(fn):
    if not os.path.isfile(fn):
        return None
    with open(fn, 'rb') as f:
        x = pkl.load(f)
    return x
read_pfl = load_pkl

def filter_df(df, func):
    # return rows for which func(index) == True
    return df[np.vectorize(func)(df.index.values)]

def get_mean_std_df(lst):
    # lst: list of identical dataframes. Return column-wise mean and std
    mean_df = copy.deepcopy(lst[0])
    std_df = copy.deepcopy(lst[0])
    for i in mean_df.index:
        for j in mean_df.columns:
            mean_df.at[i, j] = np.mean([df.at[i, j] for df in lst])
            std_df.at[i, j] = np.std([df.at[i, j] for df in lst])
    return mean_df, std_df

pretrain_rounds = {
    "emnist_resnetgn": 2000,
    "so_mini": 1000,
    "so_tiny": 1000,
    "gldv2": 2500,
    "gldv2_resnetgn": 2500
}
pretrain_ds_name = {
    "emnist_resnetgn": "emnist",
    "so_mini": "so_mini",
    "so_tiny": "so_tiny",
    "gldv2_resnetgn": "gldv2",
}

def get_fedavg_train_fn(ds_and_model, train_or_test="test"):
    return f"{OUTPUT_DIR_FL}/{pretrain_ds_name[ds_and_model]}_pretrain_{pretrain_rounds[ds_and_model]}_{train_or_test}.csv"

def get_fedavg_all_fn(ds_and_model):
    return f"{OUTPUT_DIR_FL}/{pretrain_ds_name[ds_and_model]}_pretrain_{pretrain_rounds[ds_and_model]}_test_all.p"

def get_pfl_train_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=1, train_or_test="test"):
    # state: "stateful" or "stateless"
    # train_or_test: "train" or "test"
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{train_mode}_{pfl_algo}_{init}_{state}_seed{seed}_{train_or_test}.csv"

def get_pfl_train_all_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=1):
    # per-client statistics at the end of training
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{train_mode}_{pfl_algo}_{init}_{state}_seed{seed}_test_all.p"

def get_fedavg_finetune_fn(ds_and_model, train_mode, seed=1, num_epochs_finetune=1, train_or_test="test"):
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_fedavg_{train_mode}_ne{num_epochs_finetune}_seed{seed}_{train_or_test}_finetune.csv"

def get_fedavg_finetune_all_fn(ds_and_model, train_mode, seed=1, num_epochs_finetune=1):
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_fedavg_{train_mode}_ne{num_epochs_finetune}_seed{seed}_all_finetune.p"

def get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=1, num_epochs_finetune=1, train_or_test="test"):
    if pfl_algo in ["finetune", "finetuned"]:
        return get_fedavg_finetune_fn(ds_and_model, train_mode, seed, num_epochs_finetune, train_or_test)
    else:
        return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{train_mode}_{pfl_algo}_{init}_{state}_seed{seed}_ne{num_epochs_finetune}_{train_or_test}_finetune.csv"

def get_pfl_finetune_all_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=1, num_epochs_finetune=1):
    if pfl_algo in ["finetune", "finetuned"]:
        return get_fedavg_finetune_all_fn(ds_and_model, train_mode, seed, num_epochs_finetune)
    else:
        return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{train_mode}_{pfl_algo}_{init}_{state}_seed{seed}_ne{num_epochs_finetune}_all_finetune.p"


#########################################
########### One Table Per Task
#########################################
def get_num_params():
    outs = dict(
        so_tiny=dict(
            tr_layer_0=197760,
            tr_layer_1=197760,
            adapter_16=17984,
            adapter_64=67328
        ),
        so_mini=dict(
            tr_layer_0=788736,
            tr_layer_3=788736,
            adapter_16=71808,
            adapter_64=268800
        ),
        gldv2_resnetgn=dict(
            inp_layer=9536,
            out_layer=1040364,
            adapter=1408128
        ),
        emnist_resnetgn=dict(
            inp_layer=704,
            out_layer=31806,
            adapter=1408128
        )

    )
    return outs

def get_name_list(ds_and_model):
    if ds_and_model in ["emnist_resnetgn", "gldv2_resnetgn"]:
        name_lst = ["inp_layer", "out_layer", "adapter"]
    elif ds_and_model == "so_mini":
        name_lst = ["tr_layer_0", "tr_layer_3", "adapter_16", "adapter_64"]
    elif ds_and_model == "so_tiny":
        name_lst = ["tr_layer_0", "tr_layer_1", "adapter_16", "adapter_64"]
    else:
        raise ValueError(f'Unknown ds_and_model: {ds_and_model}')
    return name_lst

def rename_one_item(name):
    if name.lower() in ["pretrain", "pretrained"]:
        return "Non-personalized"
    elif name.lower() in ["finetune", "ft"]:
        return "Finetune"
    elif name in ["inp_layer", "tr_layer_0"]:
        return "Input Layer"
    elif name in ["out_layer"] or "tr_layer" in name:
        return "Output Layer"
    elif name == "adapter":
        return "Adapter"
    elif "adapter_" in name:
        parts = name.split("_")
        return rf"Adapter (dim=${parts[1]}$)"
    else:
        raise ValueError("Unknown name:", name)

def get_main_pertask_table(
    ds_and_model, init, state, seeds=list(range(1,6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_pfl_joint=False, metric="accuracy",
):
    suffix = "_u" if use_unweighted_stats else ""
    metric_lst = [f"{metric}|mean{suffix}", f"{metric}|quantile_0.1{suffix}"]
    pfl_algo_lst = ["finetune", "pfl_alternating", "pfl_joint"]
    name_lst = get_name_list(ds_and_model)

    columns = pd.MultiIndex.from_product([metric_lst, pfl_algo_lst])
    index = ["pretrained"] + name_lst
    df_out_lst = [pd.DataFrame(columns=columns, index=index) for seed in seeds]

    for i, seed in enumerate(seeds):
        df_out = df_out_lst[i]
        fn = get_fedavg_finetune_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)
        df1 = pd.read_csv(fn, index_col=0)
        for c in columns:
            df_out.at["pretrained", c] = df1.at["pretrained", c[0]]

        for pfl_algo in pfl_algo_lst:
            ne = ne_finetune if pfl_algo == 'finetune' else ne_pfl
            if pfl_algo == "pfl_joint" and not finetune_pfl_joint:
                row = "pretrained"
            else:
                row = "finetuned"
            for train_mode in name_lst:
                fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                df2 = pd.read_csv(fn, index_col=0)
                for metric in metric_lst:
                    df_out.at[train_mode, (metric, pfl_algo)] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)
        
def get_main_pertask_table_mean(
    ds_and_model, init, state, seeds=list(range(1, 6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_pfl_joint=False
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"accuracy|mean{suffix}"
    pfl_algo_lst = ["finetune", "pfl_alternating", "pfl_joint"]
    name_lst = get_name_list(ds_and_model)

    columns = pfl_algo_lst
    index = ["pretrained"] + name_lst
    df_out_lst = [pd.DataFrame(columns=columns, index=index) for seed in seeds]

    for i, seed in enumerate(seeds):
        df_out = df_out_lst[i]
        fn = get_fedavg_finetune_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)
        df1 = pd.read_csv(fn, index_col=0)
        for c in columns:
            df_out.at["pretrained", c] = df1.at["pretrained", metric]

        for pfl_algo in pfl_algo_lst:
            ne = ne_finetune if pfl_algo == 'finetune' else ne_pfl
            if pfl_algo == "pfl_joint" and not finetune_pfl_joint:
                row = "pretrained"
            else:
                row = "finetuned"
            for train_mode in name_lst:
                fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                df2 = pd.read_csv(fn, index_col=0)
                df_out.at[train_mode, pfl_algo] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)

def rename_main_table_per_task(df):
    df = df.rename(index={name: rename_one_item(name) for name in df.index})
    df.rename(columns=metric_rename_dict, level=0, inplace=True)
    df.rename(columns=pfl_algo_rename_dict, level=1, inplace=True)
    return df

def get_final_finetune_pertask_table(
    ds_and_model, init, state, seed=1, ne_pfl=1,
    use_unweighted_stats=False,
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"accuracy|mean{suffix}"
    pfl_algo_lst = ["pfl_alternating", "pfl_joint"]
    name_lst = get_name_list(ds_and_model)

    columns = pfl_algo_lst
    index = name_lst
    df_pretrained = pd.DataFrame(columns=columns, index=index)
    df_finetuned = pd.DataFrame(columns=columns, index=index)

    for pfl_algo in pfl_algo_lst:
        for train_mode in name_lst:
            fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne_pfl)
            df2 = pd.read_csv(fn, index_col=0)
            df_pretrained.at[train_mode, pfl_algo] = df2.at["pretrained", metric]
            df_finetuned.at[train_mode, pfl_algo] = df2.at["finetuned", metric]
    return df_pretrained, df_finetuned

def rename_final_finetune_table_per_task(df):
    df = df.rename(columns=dataset_rename_dict, level=0)
    df.rename(columns=pfl_algo_rename_dict, level=1, inplace=True)
    return df

#########################################
########### Utility functions
#########################################
def convert_to_string_and_bold(
        df, dfs=None, print_std=True, do_bold=True, best_is_min = False,
        rows_to_skip = ["pretrained", "Non-personalized"]
):
    # TODO: take a per-outer_col max_or_min arg
    df2 = pd.DataFrame(index=df.index, columns=df.columns, dtype=str)  # output
    index = df.index
    outer_cols = np.unique(df.columns.get_level_values(0))
    inner_cols = np.unique(df.columns.get_level_values(1))

    # find the max values for each (row, outer_index) pairs
    df_max =  pd.DataFrame(index=index, columns=outer_cols)
    for row in index:
        for col in outer_cols:
            values = [df.at[row, (col, c)] for c in inner_cols]
            df_max.at[row, col] = min(values) if best_is_min else max(values)

    for row in index:
        for col in outer_cols:
            for c in inner_cols:
                value = df.at[row, (col, c)]
                std = dfs.at[row, (col, c)] if dfs is not None else 0
                std_str = "_{{{:.2f}}}".format(std*100) if std > 1e-4 else ""
                if do_bold and value == df_max.at[row, col] and row not in rows_to_skip:
                    cast_val = r'$\mathbf{{{:.2f}}}{}$'.format(value*100, std_str)
                else:
                    cast_val = r'${:.2f}{}$'.format(value*100, std_str)
                df2.at[row, (col, c)] = cast_val
    return df2

#######################################
### Scatter plots: per-user statistics
#######################################
def per_user_stats_scatter_plot(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)
    for i in range(n_plots):
        out = load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        ))
        test_sizes, test_metrics = out[-2:]
        out2 = load_pkl(get_fedavg_finetune_fn(ds_and_model, "finetune")) # [metrics, sizes]
        # change in accuracy
        test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, out2[0])])
        if min_is_best:
            test_deltas = -test_deltas
        test_deltas *= 100
        mean = np.average(test_deltas, weights=test_sizes)
        ax[i].scatter(test_sizes, test_deltas, alpha=0.2, **style)
        ax[i].set_xlabel("# Data per client")
        ax[i].set_ylabel("%Accuracy Change")
        ax[i].set_title(rename_one_item(name_lst[i]))
        ax[i].axhline(y=mean, color=COLORS[1], alpha=0.5, linestyle='dashed')
        ax[i].axhline(y=0, color='silver', alpha=0.5, linestyle='dotted')

    plt.tight_layout()

def per_user_fintune_stats_scatter_plot(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)
    for i in range(n_plots):
        out = load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        ))
        test_sizes, test_metrics = out[-2:]
        # change in accuracy
        test_deltas = np.asarray([t[metric_name].iloc[-1] - t[metric_name].iloc[0] for t in test_metrics])
        if min_is_best:
            test_deltas = -test_deltas
        test_deltas *= 100
        ax[i].scatter(test_sizes, test_deltas, alpha=0.2, **style)
        ax[i].set_xlabel("# Data per client")
        ax[i].set_ylabel("%Accuracy Change")
        ax[i].set_title(rename_one_item(name_lst[i]))
        ax[i].axhline(y=0, color=COLORS[1], alpha=0.5, linestyle='dashed')

    plt.tight_layout()
            
def per_user_stats_hist_plot(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)
    for i in range(n_plots):
        out = load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        ))
        test_sizes, test_metrics = out[-2:]
        # change in accuracy
        test_deltas = np.asarray([t[metric_name].iloc[-1] - t[metric_name].iloc[0] for t in test_metrics])
        if min_is_best:
            test_deltas = -test_deltas
        test_deltas *= 100
        mean = np.average(test_deltas, weights=test_sizes)
        sns.histplot(test_deltas, ax=ax[i])
        ax[i].set_yscale('log')
        ax[i].set_ylabel("#Clients")
        ax[i].set_xlabel("%Accuracy Change")
        ax[i].set_title(rename_one_item(name_lst[i]))
        ax[i].axvline(x=mean, color=COLORS[1], alpha=0.5, linestyle='dashed')
        ax[i].axvline(x=0, color='silver', alpha=0.5, linestyle='dotted')

    plt.tight_layout()