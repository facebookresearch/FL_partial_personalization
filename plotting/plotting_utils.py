# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import math
import numpy as np
import pandas as pd
import pickle as pkl
import os, sys, time, copy
import itertools
from collections import OrderedDict
from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
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
# pfl_name = r"$\partial$PFL"
pfl_name = "Partial"

pfl_algo_rename_dict = dict(
    ft = "Finetune",
    finetune = "Finetune",
    pfl_fl = "Finetune",
    pfl_am = "FedAlt",
    pfl_su = "FedSim",
    fedalt = 'FedAlt',
    fedsim = 'FedSim'
)

dataset_rename_dict = dict(
    emnist_resnetgn= "EMNIST (ResNet)",
    # gldv2_resnetgn="GLDv2 (ResNet)",
    gldv2b_resnetgn="GLDv2 (ResNet)",
    gldv2c_resnetgn="GLDv2c (ResNet)",
    so_mini="StackOverflow (Transformer)",
    so_tiny="StackOverflow (Transformer tiny)"
)

dataset_rename_dict_short = dict(
    emnist_resnetgn= "EMNIST",
    # gldv2_resnetgn="GLDv2",
    gldv2b_resnetgn="GLDv2",
    # gldv2c_resnetgn="GLDv2c",
    so_mini="StackOverflow",
    so_tiny="StackOverflow (tiny)"
)

metric_short_rename_dict = {
    "accuracy": "Accuracy",
    "loss": "Loss"
}

metric_rename_dict = {
    "accuracy|mean": "Mean Accuracy \%",
    "accuracy|std": "Std. of Accuracy \%",
    "accuracy|quantile_0.1": "10th Percentile Accuracy \%",
    "accuracy|quantile_0.25": "25th Percentile Accuracy \%",
    "accuracy|quantile_0.5": "Median Accuracy \%",
    "loss|mean": "Mean Loss",
}
for k in [1, 3, 5, 10]:
    metric_short_rename_dict[f"accuracy_top{k}"] = f"Top-{k} Accuracy"
    metric_rename_dict.update({
        f"accuracy_top{k}|mean": f"Mean Top-{k} Accuracy \%",
        f"accuracy_top{k}|std": f"Std. of Top-{k} Accuracy \%",
        f"accuracy_top{k}|quantile_0.1": f"10th Percentile Top-{k} Accuracy \%",
        f"accuracy_top{k}|quantile_0.25": f"25th Percentile Top-{k} Accuracy \%",
        f"accuracy_top{k}|quantile_0.5": f"Median Top-{k} Accuracy \%"
    })

zero_params = dict(color='black', linestyle='dotted')
mean_params = dict(color=COLORS[1], linestyle='dashed')
zero_params_hist = dict(color='black', linestyle='dotted', linewidth=1)
mean_params_hist = dict(color=COLORS[6], linestyle='dashed', linewidth=2)


from socket import gethostname
if 'devfair' in gethostname():  # fair cluster
    MAIN_DIR = '/checkpoint/pillutla/pfl'
elif 'pillutla-mbp' in gethostname(): # laptop
    MAIN_DIR = '/Users/pillutla/Dropbox (Facebook)/code/pfl_outputs'
else:
    raise ValueError('Unknown host')

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

def get_pfl_finetune_fn(
    ds_and_model, train_mode, pfl_algo, init, state, seed=1, num_epochs_finetune=1, train_or_test="test",
    reg_param=0, dropout=0
):
    if pfl_algo in ["finetune", "finetuned"]:
        return get_fedavg_finetune_fn(ds_and_model, train_mode, seed, num_epochs_finetune, train_or_test)
    else:
        if reg_param != 0:
            reg = f'reg{reg_param}_'
        elif dropout !=0:
            reg = f'do{dropout}_'
        else:
            reg = ''
        return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{train_mode}_{pfl_algo}_{reg}{init}_{state}_seed{seed}_ne{num_epochs_finetune}_{train_or_test}_finetune.csv"

def get_pfl_finetune_all_fn(
    ds_and_model, train_mode, pfl_algo, init, state, seed=1, num_epochs_finetune=1,
    reg_param=0, dropout=0
):
    if pfl_algo in ["finetune", "finetuned"]:
        return get_fedavg_finetune_all_fn(ds_and_model, train_mode, seed, num_epochs_finetune)
    else:
        if reg_param != 0:
            reg = f'reg{reg_param}_'
        elif dropout !=0:
            reg = f'do{dropout}_'
        else:
            reg = ''
        return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{train_mode}_{pfl_algo}_{reg}{init}_{state}_seed{seed}_ne{num_epochs_finetune}_all_finetune.p"

# Competing algos
def get_ditto_finetune_fn(ds_and_model, seed=1, num_epochs_finetune=1, train_or_test="test"):
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_ditto_ne{num_epochs_finetune}_seed{seed}_{train_or_test}_finetune.csv"

def get_ditto_finetune_all_fn(ds_and_model, seed=1, num_epochs_finetune=1):
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_ditto_ne{num_epochs_finetune}_seed{seed}_all_finetune.p"

def get_pfedme_finetune_fn(ds_and_model, seed=1, num_epochs_finetune=1, train_or_test="test"):
    if ds_and_model == 'so_mini':
        pfedme = 'pfedme'
    else:
        pfedme = 'pfedme2'
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{pfedme}_seed{seed}_ne{num_epochs_finetune}_{train_or_test}_finetune.csv"

def get_pfedme_finetune_all_fn(ds_and_model, seed=1, num_epochs_finetune=1):
    if ds_and_model == 'so_mini':
        pfedme = 'pfedme'
    else:
        pfedme = 'pfedme2'
    return f"{OUTPUT_DIR_PFL}/{ds_and_model}_{pfedme}_seed{seed}_ne{num_epochs_finetune}_all_finetune.p"


#########################################
########### Utils
#########################################
def get_num_params():
    outs = dict(
        so_tiny=dict(
            tr_layer_0=197760,
            tr_layer_1=197760,
            adapter_16=17984,
            adapter_64=67328,
            full=1678592
        ),
        so_mini=dict(
            tr_layer_0=788736,
            tr_layer_3=788736,
            adapter_16=71808,
            adapter_64=268800,
            full= 5721088
        ),
        gldv2b_resnetgn=dict(
            inp_layer=9536,
            out_layer=1040364,
            adapter=1408128,
            full=12216876
        ),
        emnist_resnetgn=dict(
            inp_layer=704,
            out_layer=31806,
            adapter=1408128,
            full=11199486
        )

    )
    return outs

def get_name_list(ds_and_model):
    if ds_and_model in ["emnist_resnetgn", "gldv2_resnetgn", "gldv2b_resnetgn", "gldv2c_resnetgn"]:
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

#########################################
########### Main per-task table
#########################################

def get_main_pertask_table_mean_10p(
    ds_and_model, init, state, seeds=list(range(1,6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_fedsim=False, metric_name="accuracy",
):
    suffix = "_u" if use_unweighted_stats else ""
    metric_lst = [f"{metric_name}|mean{suffix}", f"{metric_name}|quantile_0.1{suffix}"]
    pfl_algo_lst = ["finetune", "fedalt", "fedsim"]
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
            if pfl_algo == "fedsim" and not finetune_fedsim:
                row = "pretrained"
            else:
                row = "finetuned"
            for train_mode in name_lst:
                fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                df2 = pd.read_csv(fn, index_col=0)
                for metric in metric_lst:
                    df_out.at[train_mode, (metric, pfl_algo)] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)

def get_main_pertask_table_mean_states(
    ds_and_model, init, seeds=list(range(1,6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_fedsim=False, metric_name="accuracy",
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"{metric_name}|mean{suffix}"
    state_lst = ['stateful', 'stateless']
    pfl_algo_lst = ["finetune", "fedalt", "fedsim"]
    name_lst = get_name_list(ds_and_model)

    columns = pd.MultiIndex.from_product([state_lst, pfl_algo_lst])
    # index = ["pretrained"] + name_lst
    index = name_lst
    df_out_lst = [pd.DataFrame(columns=columns, index=index) for _ in seeds]

    for i, seed in enumerate(seeds):
        df_out = df_out_lst[i]
        # fn = get_fedavg_finetune_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)
        # df1 = pd.read_csv(fn, index_col=0)
        # for c in columns:
        #     df_out.at["pretrained", c] = df1.at["pretrained", metric]

        for pfl_algo in pfl_algo_lst:
            ne = ne_finetune if pfl_algo == 'finetune' else ne_pfl
            if pfl_algo == "fedsim" and not finetune_fedsim:
                row = "pretrained"
            else:
                row = "finetuned"
            for train_mode in name_lst:
                for state in state_lst:
                    fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                    df2 = pd.read_csv(fn, index_col=0)
                    df_out.at[train_mode, (state, pfl_algo)] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)

def get_main_combinedtask_table_mean(
    init="pretrained", state="stateful", seeds=list(range(1,6)), ne_finetune=5, ne_pfl=5, 
    use_unweighted_stats=False, finetune_fedsim=True, metric_name="accuracy", use_emnist=False
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"{metric_name}|mean{suffix}"
    ds_and_model_lst = ['so_mini', 'gldv2b_resnetgn']
    if use_emnist:
        ds_and_model_lst += ['emnist_resnetgn']
    pfl_algo_lst = ["finetune", "fedalt", "fedsim"]
    name_lsts = [get_name_list(ds_and_model)[:3] for ds_and_model in ds_and_model_lst]
    name_rename_dict = OrderedDict([
        ('inp_layer', 'Input Layer'), ('tr_layer_0', 'Input Layer'),
        ('out_layer', 'Output Layer'), ('tr_layer_3', 'Output Layer'),
        ('adapter', 'Adapter'), ('adapter_16', 'Adapter'), ('adapter_64', 'Adapter')
    ])
    index = [name_rename_dict[n] for n in name_lsts[1]]  # for GLDv2
    columns = pd.MultiIndex.from_product([ds_and_model_lst, pfl_algo_lst])
    df_out_lst = [pd.DataFrame(columns=columns, index=index) for _ in seeds]

    for i, seed in enumerate(seeds):
        df_out = df_out_lst[i]
        for pfl_algo in pfl_algo_lst:
            ne = ne_finetune if pfl_algo == 'finetune' else ne_pfl
            if pfl_algo == "fedsim" and not finetune_fedsim:
                row = "pretrained"
            else:
                row = "finetuned"
            for j, ds_and_model in enumerate(ds_and_model_lst):
                for train_mode in name_lsts[j]:
                    fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                    df2 = pd.read_csv(fn, index_col=0)
                    df_out.at[name_rename_dict[train_mode], (ds_and_model, pfl_algo)] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)

def get_main_pertask_table_mean(
    ds_and_model, init, state, seeds=list(range(1, 6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_fedsim=False, metric_name="accuracy"
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"{metric_name}|mean{suffix}"
    pfl_algo_lst = ["finetune", "fedalt", "fedsim"]
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
            if pfl_algo == "fedsim" and not finetune_fedsim:
                row = "pretrained"
            else:
                row = "finetuned"
            for train_mode in name_lst:
                fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                df2 = pd.read_csv(fn, index_col=0)
                df_out.at[train_mode, pfl_algo] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)

def rename_multilevel_table(df, rename_index=True):
    if rename_index:
        df = df.rename(index={name: rename_one_item(name) for name in df.index})
    else:
        df = df.copy()
    df.rename(columns=metric_rename_dict, level=0, inplace=True)  # If top level is a metric
    df.rename(columns=dataset_rename_dict_short, level=0, inplace=True)  # If top level is a dataset
    df.rename(columns={'stateless':'Stateless', 'stateful': 'Stateful'},  # If top level is state
              level=0, inplace=True)  # If top level is stateful/stateless
    df.rename(columns=pfl_algo_rename_dict, level=1, inplace=True)  # Lower level is always PFL algo
    return df

def get_final_finetune_pertask_table(
    ds_and_model, init, state, seeds=list(range(1, 6)), ne_pfl=1,
    use_unweighted_stats=False,
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"accuracy|mean{suffix}"
    pfl_algo_lst = ["fedalt", "fedsim"]
    name_lst = get_name_list(ds_and_model)

    columns = pfl_algo_lst
    index = name_lst
    df_out_lst = []

    for i, seed in enumerate(seeds):
        df_pretrained = pd.DataFrame(columns=columns, index=index)
        df_finetuned = pd.DataFrame(columns=columns, index=index)
        for pfl_algo in pfl_algo_lst:
            for train_mode in name_lst:
                fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne_pfl)
                df2 = pd.read_csv(fn, index_col=0)
                df_pretrained.at[train_mode, pfl_algo] = df2.at["pretrained", metric]
                df_finetuned.at[train_mode, pfl_algo] = df2.at["finetuned", metric]
        df_out_lst.append(df_finetuned - df_pretrained)
    return get_mean_std_df(df_out_lst)

def rename_final_finetune_table_per_task(df):
    df = df.rename(columns=dataset_rename_dict_short, level=0)
    df.rename(columns=pfl_algo_rename_dict, level=1, inplace=True)
    return df

#########################################
########### Main per-task table
#########################################
def get_main_combined_table(
    ds_and_model_lst, pfl_algo, init, state, seeds=list(range(1, 6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, metric_name='accuracy', metric_name_dict={}
):
    suffix = "_u" if use_unweighted_stats else ""

    df_mean = {}
    df_std = {}
    for ds_and_model in ds_and_model_lst:
        if ds_and_model in metric_name_dict:
            metric = f"{metric_name_dict[ds_and_model]}|mean{suffix}"
        else:
            metric = f"{metric_name}|mean{suffix}"
        name_lst = get_name_list(ds_and_model)[:3]  # [Inp, Out, Adapter]
        col_lst = []
        for i, seed in enumerate(seeds):
            col = OrderedDict()
            # FedAvg
            fn = get_fedavg_finetune_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)
            df1 = pd.read_csv(fn, index_col=0)
            col[("Non-personalized", " ")] = df1.at["pretrained", metric]
            col[("Full Model", "Finetune")] = df1.at["finetuned", metric]
            # Ditto
            fn = get_ditto_finetune_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)
            df1 = pd.read_csv(fn, index_col=0)
            col[("Full Model", "Ditto")] = df1.at["finetuned", metric]
            # pFedMe
            fn = get_pfedme_finetune_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)
            df1 = pd.read_csv(fn, index_col=0)
            col[("Full Model", "pFedMe")] = df1.at["finetuned", metric]
            # PFL
            for train_mode in name_lst:
                fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne_pfl)
                df1 = pd.read_csv(fn, index_col=0)
                tm = "adapter" if "adapter_" in train_mode else train_mode
                col[(pfl_name, rename_one_item(tm))] = df1.at["finetuned", metric]
            col_lst.append(col)
        df =  pd.DataFrame(col_lst).T  # seeds as columns, expected index as index
        df_mean[dataset_rename_dict_short[ds_and_model]] = df.mean(axis=1)
        df_std[dataset_rename_dict_short[ds_and_model]] = df.std(axis=1)
    df_mean, df_std = pd.DataFrame(df_mean), pd.DataFrame(df_std)
    for dfx in [df_mean, df_std]:
        dfx.index = pd.MultiIndex.from_tuples(dfx.index)
    return df_mean, df_std


#########################################
########### Utility functions
#########################################
def convert_to_string_and_bold_multiindexcol(
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
                if dfs is not None:
                    std = dfs.at[row, (col, c)]
                    std_str = "_{{{:.2f}}}".format(std*100)
                else:
                    std_str = "_{}"
                if do_bold and value == df_max.at[row, col] and row not in rows_to_skip:
                    cast_val = r'$\mathbf{{{:.2f}}}{}$'.format(value*100, std_str)
                else:
                    cast_val = r'${:.2f}{}$'.format(value*100, std_str)
                df2.at[row, (col, c)] = cast_val
    return df2

def convert_to_string_and_bold_maxpercol(
        df, dfs=None, print_std=True, do_bold=True, best_is_min = False
):
    # TODO: take a per-outer_col max_or_min arg
    df_max = df.min(axis=0) if best_is_min else df.max(axis=0)
    df2 = pd.DataFrame(index=df.index, columns=df.columns, dtype=str)  # output

    for row in df.index:
        for col in df.columns:
            value = df.at[row, col]
            if dfs is not None:
                std = dfs.at[row, col]
                std_str = "_{{{:.2f}}}".format(std*100)
            else:
                std_str = "_{}"
            if do_bold and value == df_max[col]:
                cast_val = r'$\mathbf{{{:.2f}}}{}$'.format(value*100, std_str)
            else:
                cast_val = r'${:.2f}{}$'.format(value*100, std_str)
            df2.at[row, col] = cast_val
    return df2

#######################################
### Scatter plots: per-user statistics
#######################################

def _scatter_plot_helper_1(
    f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
    clean_name_lst, metric_name, train_or_test, x_train_or_test, min_is_best, style,
    hist2d, xlim, ylim, gridsize, annotate=True
):
    n_plots = ax.shape[0]
    color_rgb = mpl.colors.colorConverter.to_rgb(COLORS[0])
    colors = [sns.utils.set_hls_values(color_rgb, l=l)  # noqa
            for l in np.linspace(0.99, 0, 12)]
    cmap = sns.blend_palette(colors, as_cmap=True)
    # cmap = 'Pastel1'
    normalizer = None
    lst = []
    for r in range(2):
        for i in range(n_plots):
            train_sizes, train_metrics, test_sizes, test_metrics = outs[i]
            # change in loss/accuracy
            if train_or_test == 'train':
                test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(train_metrics, train_metrics_fedavg)])
            else:
                test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, test_metrics_fedavg)])
            if min_is_best:
                test_deltas = -test_deltas
            if 'accuracy' in metric_name:
                test_deltas *= 100
            mean = np.average(test_deltas, weights=test_sizes)
            xs = np.asarray(train_sizes) if x_train_or_test.lower() == 'train' else np.asarray(test_sizes)
            if r == 0:
                print(f'i = {i}\tmean={mean:.4f}\tgt0={(test_deltas>0).sum()}\tlt0={(test_deltas<0).sum()}\teq0={(test_deltas==0).sum()}')
            if hist2d:
                ax[i].axhline(y=mean, alpha=0.6, **mean_params_hist, label='mean' if r==0 else None)
                ax[i].axhline(y=0, alpha=0.4, **zero_params_hist)
                idxs1 = (xs != -1e12) if xlim is None else np.logical_and(xlim[0] <= xs, xs <= xlim[1])
                idxs2 = (xs != -1e12) if ylim is None else np.logical_and(ylim[0] <= test_deltas, test_deltas <= ylim[1])
                idxs = np.logical_and(idxs1, idxs2)  # all the points within the given limits
                out = ax[i].hexbin(
                    xs[idxs], test_deltas[idxs], gridsize=gridsize[i] if isinstance(gridsize, Sequence) else gridsize, 
                    cmap=cmap, norm=normalizer
                )
                lst.extend(out.get_array().tolist())
            else:
                ax[i].axhline(y=mean, alpha=0.5, **mean_params, label='mean' if r==0 and annotate else None)
                ax[i].axhline(y=0, alpha=0.5, **zero_params)
                ax[i].scatter(xs, test_deltas, alpha=0.1, **style)
            # ax[i].set_xlabel(f"# Data per client ({x_train_or_test})")
            ax[i].set_xlabel(f"# Data per device")
            if i==0 and annotate: ax[i].set_ylabel(r"$\Delta$ " + metric_short_rename_dict[metric_name])
            ax[i].set_title(clean_name_lst[i])
            if ylim is not None:
                ax[i].set_ylim(ylim)
            if xlim is not None:
                ax[i].set_xlim(xlim)
        if hist2d: # set norm
            normalizer=Normalize(min(lst), max(lst))
            im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
        else:  # no need to repeat twice for a scatter plot
            break 
    if annotate: ax[-1].legend(fontsize=15)
    plt.tight_layout()
    if annotate:
        ylim = ax[0].get_ylim()
        xlim = ax[0].get_xlim()
        _y = (ylim[1] - ylim[0]) * 0.01
        _x = 0.2 * xlim[0] + 0.8 * xlim[1]
        ax[0].annotate("Pers. helps", xy=(_x, _y),  xytext=(_x, 0.8 * ylim[1]), 
            xycoords='data', textcoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"), fontsize=12,
            ha='center', arrowprops=dict(arrowstyle="<|-"))
        ax[0].annotate("Pers. hurts", xy=(_x, -_y),  xytext=(_x, 0.8 * ylim[0]), 
            xycoords='data', textcoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"), fontsize=12,
            ha='center', arrowprops=dict(arrowstyle="<|-"))
    if hist2d:
        cb = f.colorbar(im, ax=ax.ravel().tolist())
        extra_artists = (cb,)
    else:
        extra_artists = ()
    print('-'*50)
    return extra_artists
            


def per_user_stats_scatter_plot(ds_and_model, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
        hist2d=False, xlim=None, ylim=None, gridsize=25
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)[:3]
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), sharey=True)
    # suptitle = f.suptitle(dataset_rename_dict_short[ds_and_model], fontsize=18)
    outs = [
        load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        ))
        for i in range(n_plots)
    ]
    out2 = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    test_metrics_fedavg = out2[-1]
    train_metrics_fedavg = out2[1]
    clean_name_lst = [rename_one_item(name) for name in name_lst]
    extra_artists = _scatter_plot_helper_1(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        clean_name_lst, metric_name, train_or_test, x_train_or_test, min_is_best, style, 
        hist2d, xlim, ylim, gridsize
    )
    # extra_artists = (suptitle, *extra_artists)
    return f, extra_artists

def per_user_stats_scatter_plot_full_v_partial(ds_and_model, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
        ne_finetune=5, hist2d=False, xlim=None, ylim=None, gridsize=25
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    n_plots = 4
    f, ax = plt.subplots(1, n_plots, figsize=(4.5*n_plots, 4), sharey=True)
    # suptitle = f.suptitle(dataset_rename_dict_short[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "tr_layer_3"
    outs = [
        load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", num_epochs_finetune=ne_finetune)),  # FedAvg + Full finetune
        load_pkl(get_ditto_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # Ditto
        load_pkl(get_pfedme_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # pFedMe
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, **args))  # PFL
    ]
    clean_name_lst = ["Finetune", "Ditto", "pFedMe", pfl_name]
    _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0]
    extra_artists = _scatter_plot_helper_1(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        clean_name_lst, metric_name, train_or_test, x_train_or_test, min_is_best, style, 
        hist2d, xlim, ylim, gridsize
    )
    # extra_artists = (suptitle, *extra_artists) 
    return f, extra_artists

def per_user_stats_scatter_plot_full_v_partial_main(ds_and_model, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train',
        ne_finetune=5, hist2d=False, xlim=None, ylim=None, gridsize=25
):
    if hist2d:
        raise ValueError('Split train-test does not work with hist2d')
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    n_plots = 4
    f, ax = plt.subplots(1, n_plots, figsize=(4.5*n_plots, 4), sharey=False)
    ax = np.asarray([ax[0], ax[2], ax[1], ax[3]])
    ax[0].sharey(ax[1]); ax[2].sharey(ax[3])  # share axes
    # suptitle = f.suptitle(dataset_rename_dict_short[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "tr_layer_3"
    outs = [
        load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", num_epochs_finetune=ne_finetune)),  # FedAvg + Full finetune
        # load_pkl(get_ditto_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # Ditto
        # load_pkl(get_pfedme_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # pFedMe
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, **args))  # PFL
    ]
    clean_name_lst = ["Full (train)", "Partial (train)", "Full (test)", "Partial (test)"]
    _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0]
    extra_artists_tr = _scatter_plot_helper_1(
        f, ax[:2], outs, train_metrics_fedavg, test_metrics_fedavg, 
        clean_name_lst[:2], metric_name, 'train', x_train_or_test, min_is_best, style, 
        hist2d, xlim, ylim[0], gridsize, annotate=True
    )
    extra_artists_te = _scatter_plot_helper_1(
        f, ax[2:], outs, train_metrics_fedavg, test_metrics_fedavg, 
        clean_name_lst[2:], metric_name, 'test', x_train_or_test, min_is_best, style, 
        hist2d, xlim, ylim[1], gridsize, annotate=False
    )
    # extra_artists = (suptitle, *extra_artists) 
    extra_artists = (*extra_artists_tr, *extra_artists_te)
    return f, extra_artists


def per_user_stats_scatter_plot_regularization(ds_and_model, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
        hist2d=False, xlim=None, ylim=None, gridsize=25
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    if 'emnist' in ds_and_model:
        reg_list = [0, 0.1, 100]
    elif 'so_mini' in ds_and_model:
        reg_list = [0, 0.001, 10]
    elif 'gldv2' in ds_and_model:
        reg_list = [0, 0.1, 100]
    else:
        raise ValueError(f'Unknown ds_and_model {ds_and_model}')
    n_plots = len(reg_list)
    f, ax = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), sharey=True)
    # suptitle = f.suptitle(dataset_rename_dict_short[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, reg_param=reg, **args))
        for reg in reg_list
    ]
    clean_name_lst = [f"Reg. Param. = {reg}" for reg in ['0', 'best', 'large']]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) 
    extra_artists = _scatter_plot_helper_1(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        clean_name_lst, metric_name, train_or_test, x_train_or_test, min_is_best, style, 
        hist2d, xlim, ylim, gridsize
    )
    # extra_artists = (suptitle, *extra_artists) 
    return f, extra_artists

def per_user_stats_scatter_plot_dropout(ds_and_model, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
        hist2d=False, xlim=None, ylim=None, gridsize=25
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    dropout_list = [0, 0.3, 0.7]
    n_plots = len(dropout_list)
    f, ax = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), sharey=True)
    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "tr_layer_3"  # TODO!
    outs = [
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, dropout=dropout, **args))
        for dropout in dropout_list
    ]
    clean_name_lst = [f"Dropout = {dropout}" for dropout in [0, 'best', 'large']]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune"))
    extra_artists = _scatter_plot_helper_1(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        clean_name_lst, metric_name, train_or_test, x_train_or_test, min_is_best, style, 
        hist2d, xlim, ylim, gridsize
    )
    return f, extra_artists

def _boxplot_helper_3(outs, train_metrics_fedavg, test_metrics_fedavg, ax, clean_name_lst, metric_name, 
                      train, min_is_best, boxplot, rotation, annotate):
    ax.axhline(y=0, alpha=0.5, **zero_params)
    n_plots = len(clean_name_lst)
    data = {}
    for i in range(n_plots):
        train_sizes, train_metrics, test_sizes, test_metrics = outs[i]
        # change in accuracy
        if train:
            test_deltas = [100*(t[metric_name].iloc[-1] - t1[metric_name].iloc[0]) for t, t1 in zip(train_metrics, train_metrics_fedavg)]
        else:
            test_deltas = [100*(t[metric_name].iloc[-1] - t1[metric_name].iloc[0]) for t, t1 in zip(test_metrics, test_metrics_fedavg)]
        if min_is_best:
            test_deltas = -test_deltas
        data[clean_name_lst[i]] = test_deltas
    data = pd.DataFrame(data)
    if boxplot:
        sns.boxplot(data=data, ax=ax)
    else:
        sns.violinplot(data=data, ax=ax)
    ax.set_yticklabels([rf'${round(a)}$' for a in ax.get_yticks()], size=12)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=rotation, size=18)
    ax.set_ylabel(r'$\Delta$ Accuracy', fontsize=18)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([xlim[0], xlim[1]], [0, 0], [ylim[1], ylim[1]], color=COLORS[-1], alpha=0.1, zorder=-1)
    ax.fill_between([xlim[0], xlim[1]], [0, 0], [ylim[0], ylim[0]], color=COLORS[3], alpha=0.1, zorder=-1)
    if annotate:
        ax.text(0.63 * xlim[1], 0.85 * ylim[1], "Pers. helps", fontdict=dict(fontsize=15),
                bbox=dict(boxstyle="round", fc="none", ec="gray"))
        ax.text(0.63 * xlim[1], 0.88 * ylim[0], "Pers. hurts", fontdict=dict(fontsize=15),
                bbox=dict(boxstyle="round", fc="none", ec="gray"))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def per_user_stats_scatter_plot_3(ds_and_model, ax=None, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', train=False, min_is_best=False, boxplot=False, rotation=0, annotate=True
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
    name_lst = get_name_list(ds_and_model)[:3]
    n_plots = len(name_lst)
    ax.set_title(dataset_rename_dict_short[ds_and_model], fontsize=18)
    outs = [
        load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        )) for i in range(n_plots)
    ]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    clean_name_lst = ["Adapter" if 'adapter' in name else rename_one_item(name) for name in name_lst]
    _boxplot_helper_3(outs, train_metrics_fedavg, test_metrics_fedavg, ax, clean_name_lst, metric_name, train, min_is_best, boxplot, rotation, annotate)

def per_user_scatter_plot_3_full_v_partial(ds_and_model, ax=None, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', train=False, min_is_best=False, 
        boxplot=False, rotation=0, ne_finetune=5, annotate=True
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.set_title('Per-device Statistics ({})'.format('train' if train else 'test'))
    name_lst = get_name_list(ds_and_model) # PFL names
    # pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    pfl_model_name = "out_layer" if "out_layer" in name_lst else "tr_layer_3"
    outs = [
        load_pkl(get_ditto_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # Ditto
        load_pkl(get_pfedme_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # pFedMe
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, **args))  # PFL
    ]
    clean_name_lst = ["Ditto", "pFedMe", pfl_name]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    _boxplot_helper_3(outs, train_metrics_fedavg, test_metrics_fedavg, ax, clean_name_lst, metric_name, train, min_is_best, boxplot, rotation, annotate)

def per_user_stats_scatter_plot_3_regularization(ds_and_model, ax=None, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', train=False, min_is_best=False, boxplot=False, rotation=0, annotate=True
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
    if 'emnist' in ds_and_model:
        reg_list = [0, 0.1, 100]
    elif 'so_mini' in ds_and_model:
        reg_list = [0, 0.001, 10]
    elif 'gldv2' in ds_and_model:
        reg_list = [0, 0.1, 100]
    else:
        raise ValueError(f'Unknown ds_and_model {ds_and_model}')
    n_plots = len(reg_list)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, reg_param=reg, **args))
        for reg in reg_list
    ]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    clean_name_lst = ["No Reg.", "Best Reg.", "Large Reg."]
    ax.set_title('Effect of Regularization ({})'.format('train' if train else 'test'), fontsize=18)
    _boxplot_helper_3(outs, train_metrics_fedavg, test_metrics_fedavg, ax, clean_name_lst, metric_name, train, min_is_best, boxplot, rotation, annotate)

def per_user_stats_scatter_plot_3_dropout(ds_and_model, ax=None, pfl_algo="fedalt", 
        args={}, metric_name='accuracy', train=False, min_is_best=False, boxplot=False, rotation=0, annotate=True
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
    if 'emnist' in ds_and_model:
        do_list = [0, 0.3, 0.7]
        pfl_model_name = 'adapter'
    elif 'so_mini' in ds_and_model:
        do_list = [0, 0.3, 0.7]
        pfl_model_name = 'tr_layer_3'
    elif 'gldv2' in ds_and_model:
        do_list = [0, 0.3, 0.7]
        pfl_model_name = 'adapter'
    else:
        raise ValueError(f'Unknown ds_and_model {ds_and_model}')
    n_plots = len(do_list)

    # Gather filenames
    outs = [
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, dropout=dropout, **args))
        for dropout in do_list
    ]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    clean_name_lst = ["No d/o", "Best d/o", "Large d/o"]
    ax.set_title('Effect of Dropout ({})'.format('train' if train else 'test'), fontsize=18)
    _boxplot_helper_3(outs, train_metrics_fedavg, test_metrics_fedavg, ax, clean_name_lst, metric_name, train, min_is_best, boxplot, rotation, annotate)
            
def _scatter_plot_helper_4(
    f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
    seeds, clean_name_lst, metric_name, min_is_best, ylims, ylim, train, annotate
):
    n_plots = len(ax)
    for i in range(n_plots):
        test_deltas_lst = []
        for j, seed in enumerate(seeds):
            train_sizes, train_metrics, test_sizes, test_metrics = outs[j][i]  
            # change in accuracy
            if train:
                test_deltas = [t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(train_metrics, train_metrics_fedavg)]
            else:
                test_deltas = [t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, test_metrics_fedavg)]
            test_deltas_lst.append(test_deltas)
        test_deltas = np.stack(test_deltas_lst)  # (seed, client)
        if min_is_best:
            test_deltas = -test_deltas
        if 'accuracy' in metric_name:
            test_deltas *= 100
        mean_per_client = np.mean(test_deltas, axis=0)
        std_per_client = np.std(test_deltas, axis=0)
        max_per_client = np.max(test_deltas, axis=0)
        min_per_client = np.min(test_deltas, axis=0)
        idxs = np.argsort(mean_per_client)
        xs = np.arange(idxs.shape[0])+1
        mean_per_client, std_per_client = mean_per_client[idxs], std_per_client[idxs]
        max_per_client, min_per_client = max_per_client[idxs], min_per_client[idxs]
        idx1 = np.argmax((max_per_client - min_per_client) * ((mean_per_client < 0) & (max_per_client > 0) & (min_per_client < 0)).astype(np.float64))  # negative mean
        idx2 = np.argmax((max_per_client - min_per_client) * ((mean_per_client > 0) & (max_per_client > 0) & (min_per_client < 0)).astype(np.float64))  # positive mean
        ax[i].plot(xs, mean_per_client, linestyle='solid', label='mean per device' if i==0 else None)
        ax[i].fill_between(xs, max_per_client, min_per_client, alpha=0.6, label='max/min per device' if i==0 else None)
        ax[i].set_xlabel('Device Rank')
        ax[i].set_ylabel(r'$\Delta$ ' + metric_short_rename_dict[metric_name])
        ax[i].set_title(clean_name_lst[i])
        ax[i].axhline(y=0, alpha=0.5, **zero_params)
        if annotate:
            idxs = np.asarray([idx1, idx2])
            yerr = np.stack([mean_per_client[idxs] - min_per_client[idxs], max_per_client[idxs] - mean_per_client[idxs]], axis=0)
            # print(idxs, mean_per_client[idxs], min_per_client[idxs], max_per_client[idxs])
            ax[i].scatter(
                idxs, mean_per_client[idxs],
                marker='o', color=COLORS[1], s=100, alpha=0.5, zorder=10
            )
            ax[i].errorbar(
                idxs, mean_per_client[idxs], yerr=yerr,
                fmt='none', color=COLORS[1], markersize=12, alpha=0.5, zorder=11, capsize=5, capthick=3
            )
        if ylims is not None and len(ylims) == n_plots:
            ylim = ylims[i]
        if ylim is not None:
            ax[i].set_ylim(ylim)
    if annotate:
        lgd = f.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2)
        extra_artists = (lgd,)
        ylim = ax[0].get_ylim()
        xlim = ax[0].get_xlim()
        _x = 0.3 * xlim[0] + 0.7 * xlim[1]
        ax[0].annotate("Pers. helps", xy=(_x, 0),  xytext=(_x, 0.8 * ylim[1]), 
            xycoords='data', textcoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"), fontsize=12,
            ha='center', arrowprops=dict(arrowstyle="<|-"))
        ax[0].annotate("Pers. hurts", xy=(_x, 0),  xytext=(_x, 0.8 * ylim[0]), 
            xycoords='data', textcoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"), fontsize=12,
            ha='center', arrowprops=dict(arrowstyle="<|-"))
    else:
        extra_artists = ()
    plt.tight_layout()
    return extra_artists
        
def per_user_stats_scatter_plot_4(ds_and_model, pfl_algo="fedalt", 
        args={}, seeds=list(range(1, 6)), metric_name='accuracy', min_is_best=False, ylims=None, ylim=None, train=False,
        annotate=False, 
):
    name_lst = get_name_list(ds_and_model)[:3]
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(4.5*n_plots, 4))
    # suptitle = f.suptitle(dataset_rename_dict_short[ds_and_model], fontsize=18)
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    outs = [[
        load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, seed=seed, **args
        )) for i in range(n_plots)] for seed in seeds
    ]
    clean_name_lst = [rename_one_item(name_lst[i]) for i in range(n_plots)]
    _scatter_plot_helper_4(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        seeds, clean_name_lst, metric_name, min_is_best, ylims, ylim, train, annotate
    )
    # extra_artists = (suptitle,)
    extra_artists = ()
    return f, extra_artists

def per_user_stats_scatter_plot_full_v_partial_4(ds_and_model, seeds=list(range(1, 6)), pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, ylims=None, ylim=None, train=False,
        ne_finetune=5,  annotate=False
):
    n_plots = 4
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    # suptitle = f.suptitle(dataset_rename_dict_short[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [[
            load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)),  # FedAvg + Full finetune
            load_pkl(get_ditto_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # Ditto
            load_pkl(get_pfedme_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # pFedMe
            load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, seed=seed, **args))  # PFL
        ] for seed in seeds]
    clean_name_lst = ["Finetune", "Ditto", "pFedMe", pfl_name]
    _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0][0]

    _scatter_plot_helper_4(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        seeds, clean_name_lst, metric_name, min_is_best, ylims, ylim, train, annotate
    )
    # extra_artists = (suptitle,)
    extra_artists = ()
    return f, extra_artists


def per_user_stats_scatter_plot_full_v_partial_4_main(ds_and_model, seeds=list(range(1, 6)), pfl_algo="fedalt", 
        args={}, metric_name='accuracy', min_is_best=False, ylims=None, ylim=None, train=False,
        ne_finetune=5, annotate=True
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    n_plots = 2
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "tr_layer_3"
    outs = [[
            load_pkl(get_ditto_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # Ditto
            load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, seed=seed, **args))  # PFL
        ] for seed in seeds]
    clean_name_lst = ["Ditto", pfl_name]
    _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0][0]  # Ditto + pretrained = fedavg output

    extra_artists = _scatter_plot_helper_4(
        f, ax, outs, train_metrics_fedavg, test_metrics_fedavg, 
        seeds, clean_name_lst, metric_name, min_is_best, ylims, ylim, train, annotate
    )
    return f, extra_artists

# Per client 
def per_user_sizes(ds_and_model, ax, test=False, bins='auto'):
    out = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", num_epochs_finetune=5))
    train_sizes, _, test_sizes, _ = out
    sizes = test_sizes if test else train_sizes
    sns.histplot(sizes, bins=bins, kde=True, stat='count', ax=ax, alpha=0.2)
    if ds_and_model == 'so_mini':
        suffix = " (words)"
    else:
        suffix = " (images)"
    ax.set_xlabel('# Data per device' + suffix)
    ax.set_ylabel('Count')
    ax.set_title(dataset_rename_dict_short[ds_and_model])
    # Print other stats
    median = np.median(sizes)
    m1 = np.max(sizes)
    m2 = np.min(sizes)
    m = np.mean(sizes)
    print(f'''{ds_and_model}\t median: {median}\t max: {m1}\t min: {m2}\t mean: {m}''')
