
import math
import numpy as np
import pandas as pd
import pickle as pkl
import os, sys, time, copy
import itertools
from collections import OrderedDict
from scipy.stats import spearmanr, pearsonr

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
    gldv2c_resnetgn="GLDv2c",
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


from socket import gethostname
if 'devfair' in gethostname():
    MAIN_DIR = '/checkpoint/pillutla/pfl'
elif 'pillutla-mbp' in gethostname():
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
    use_unweighted_stats=False, finetune_pfl_joint=False, metric_name="accuracy",
):
    suffix = "_u" if use_unweighted_stats else ""
    metric_lst = [f"{metric_name}|mean{suffix}", f"{metric_name}|quantile_0.1{suffix}"]
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

def get_main_pertask_table_mean_states(
    ds_and_model, init, seeds=list(range(1,6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_pfl_joint=False, metric_name="accuracy",
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"{metric_name}|mean{suffix}"
    state_lst = ['stateful', 'stateless']
    pfl_algo_lst = ["finetune", "pfl_alternating", "pfl_joint"]
    name_lst = get_name_list(ds_and_model)

    columns = pd.MultiIndex.from_product([state_lst, pfl_algo_lst])
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
                for state in state_lst:
                    fn = get_pfl_finetune_fn(ds_and_model, train_mode, pfl_algo, init, state, seed=seed, num_epochs_finetune=ne)
                    df2 = pd.read_csv(fn, index_col=0)
                    df_out.at[train_mode, (state, pfl_algo)] = df2.at[row, metric]
    return get_mean_std_df(df_out_lst)

def get_main_pertask_table_mean(
    ds_and_model, init, state, seeds=list(range(1, 6)), ne_finetune=5, ne_pfl=1,
    use_unweighted_stats=False, finetune_pfl_joint=False, metric_name="accuracy"
):
    suffix = "_u" if use_unweighted_stats else ""
    metric = f"{metric_name}|mean{suffix}"
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

def rename_multilevel_table(df):
    df = df.rename(index={name: rename_one_item(name) for name in df.index})
    df.rename(columns=metric_rename_dict, level=0, inplace=True)  # If top level is a metric
    df.rename(columns=dataset_rename_dict, level=0, inplace=True)  # If top level is a dataset
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
    pfl_algo_lst = ["pfl_alternating", "pfl_joint"]
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
    df = df.rename(columns=dataset_rename_dict, level=0)
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
            col[("Non-personalized", "")] = df1.at["pretrained", metric]
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
                col[("PFL", rename_one_item(tm))] = df1.at["finetuned", metric]
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
def per_user_stats_scatter_plot(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)
    for i in range(n_plots):
        out = load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        ))
        train_sizes, train_metrics, test_sizes, test_metrics = out
        out2 = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
        test_metrics_fedavg = out2[-1]
        train_metrics_fedavg = out2[1]
        # change in accuracy
        if train_or_test == 'train':
            test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(train_metrics, train_metrics_fedavg)])
        else:
            test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, test_metrics_fedavg)])
        if min_is_best:
            test_deltas = -test_deltas
        if 'accuracy' in metric_name:
            test_deltas *= 100
        mean = np.average(test_deltas, weights=test_sizes)
        xs = train_sizes if x_train_or_test.lower() == 'train' else test_sizes
        ax[i].scatter(xs, test_deltas, alpha=0.1, **style)
        ax[i].set_xlabel(f"# Data per client ({x_train_or_test})")
        if i==0: ax[i].set_ylabel(r"$\Delta$ " + metric_short_rename_dict[metric_name])
        ax[i].set_title(rename_one_item(name_lst[i]))
        ax[i].axhline(y=mean, alpha=0.5, **mean_params)
        ax[i].axhline(y=0, alpha=0.8, **zero_params)

    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

def per_user_stats_scatter_plot_full_v_partial(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
        ne_finetune=5
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    n_plots = 4
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [
        load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", num_epochs_finetune=ne_finetune)),  # FedAvg + Full finetune
        load_pkl(get_ditto_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # Ditto
        load_pkl(get_pfedme_finetune_all_fn(ds_and_model, num_epochs_finetune=ne_finetune)), # pFedMe
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, **args))  # PFL
    ]
    clean_name_lst = ["Finetune", "Ditto", "pFedMe", "Adapter"]
    _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0]
    
    
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
        xs = train_sizes if x_train_or_test.lower() == 'train' else test_sizes
        ax[i].scatter(xs, test_deltas, alpha=0.1, **style)
        ax[i].set_xlabel(f"# Data per client ({x_train_or_test})")
        if i==0: ax[i].set_ylabel(r"$\Delta$ " + metric_short_rename_dict[metric_name])
        ax[i].set_title(clean_name_lst[i])
        ax[i].axhline(y=mean, alpha=0.5, **mean_params)
        ax[i].axhline(y=0, alpha=0.5, **zero_params)
    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

def per_user_stats_scatter_plot_regularization(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
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
    # reg_list = [0, '1e-4', '0.001', 0.01, 0.1, '1.0']
    n_plots = len(reg_list)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, reg_param=reg, **args))
        for reg in reg_list
    ]
    clean_name_lst = [f"Reg. Param. = {reg}" for reg in reg_list]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune"))
    
    
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
        xs = train_sizes if x_train_or_test.lower() == 'train' else test_sizes
        ax[i].scatter(xs, test_deltas, alpha=0.1, **style)
        ax[i].set_xlabel(f"# Data per client ({x_train_or_test})")
        if i==0: ax[i].set_ylabel(r"$\Delta$ " + metric_short_rename_dict[metric_name])
        ax[i].set_title(clean_name_lst[i])
        ax[i].axhline(y=mean, alpha=0.5, **mean_params)
        ax[i].axhline(y=0, alpha=0.5, **zero_params)
    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

def per_user_stats_scatter_plot_dropout(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, x_train_or_test='train', train_or_test='test',
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    dropout_list = [0, 0.3, 0.5, 0.7, 0.9]
    n_plots = len(dropout_list)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "tr_layer_3"  # TODO!
    outs = [
        load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, dropout=dropout, **args))
        for dropout in dropout_list
    ]
    clean_name_lst = [f"Dropout = {dropout}" for dropout in dropout_list]
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune"))
    
    
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
        xs = train_sizes if x_train_or_test.lower() == 'train' else test_sizes
        ax[i].scatter(xs, test_deltas, alpha=0.1, **style)
        ax[i].set_xlabel(f"# Data per client ({x_train_or_test})")
        if i==0: ax[i].set_ylabel(r"$\Delta$ " + metric_short_rename_dict[metric_name])
        ax[i].set_title(clean_name_lst[i])
        ax[i].axhline(y=mean, alpha=0.5, **mean_params)
        ax[i].axhline(y=0, alpha=0.5, **zero_params)
    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

def per_user_stats_scatter_plot_2(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, kdeplot=False
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
        train_sizes, _, test_sizes, test_metrics = out
        test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune"))[-1] # load the metrics only
        # change in accuracy
        test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, test_metrics_fedavg)])
        if min_is_best:
            test_deltas = -test_deltas
        if "accuracy" in metric_name:
            test_deltas *= 100
        mean = np.average(test_deltas, weights=test_sizes)
        xs = np.asarray([t1[metric_name].iloc[0] for t1 in test_metrics_fedavg]) * 100
        if kdeplot:
            sns.kdeplot(x=xs, y=test_deltas, ax=ax[i])
        else:
            ax[i].scatter(xs, test_deltas, alpha=0.1, **style)
        print(ds_and_model, i, pearsonr(xs, test_deltas))
        ax[i].set_xlabel(f"Non-pers. Accuracy %")
        ax[i].set_ylabel(r"$\Delta$ " + metric_short_rename_dict[metric_name])
        ax[i].set_title(rename_one_item(name_lst[i]))
        ax[i].axhline(y=mean, alpha=0.5, **mean_params)
        ax[i].axhline(y=0, alpha=0.8, **zero_params)

    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

def per_user_stats_scatter_plot_3(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, boxplot=False
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.set_title(dataset_rename_dict[ds_and_model], fontsize=18)
    data = {}
    for i in range(n_plots):
        out = load_pkl(get_pfl_finetune_all_fn(
            ds_and_model, name_lst[i], pfl_algo, **args
        ))
        train_sizes, _, test_sizes, test_metrics = out
        test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune"))[-1] # load the metrics only
        # change in accuracy
        test_deltas = [t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, test_metrics_fedavg)]
        if min_is_best:
            test_deltas = -test_deltas
        test_deltas *= 100
        data[name_lst[i]] = test_deltas
    data = pd.DataFrame(data)
    if boxplot:
        sns.boxplot(data=data)
    else:
        sns.violinplot(data=data)
    plt.tight_layout()
    extra_artists = ()
    return f, extra_artists
            
def per_user_stats_scatter_plot_4(ds_and_model, pfl_algo="pfl_alternating", 
        args={}, seeds=list(range(1, 6)), metric_name='accuracy', min_is_best=False, train=False
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    name_lst = get_name_list(ds_and_model)
    n_plots = len(name_lst)
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)
    _, train_metrics_fedavg, _, test_metrics_fedavg = load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune")) # load the metrics only
    for i in range(n_plots):
        test_deltas_lst = []
        for seed in seeds:
            out = load_pkl(get_pfl_finetune_all_fn(
                ds_and_model, name_lst[i], pfl_algo, seed=seed, **args
            ))
            train_sizes, train_metrics, test_sizes, test_metrics = out    
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
        ax[i].plot(xs, mean_per_client, linestyle='dashed')
        ax[i].fill_between(xs, max_per_client, min_per_client, alpha=0.6)
        ax[i].set_xlabel('Client Id')
        ax[i].set_ylabel(r'$\Delta$ ' + metric_short_rename_dict[metric_name])
        ax[i].set_title(rename_one_item(name_lst[i]))
        ax[i].axhline(y=0, alpha=0.3, **zero_params)
    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

def per_user_stats_scatter_plot_full_v_partial_4(ds_and_model, seeds=list(range(1, 6)), pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, train=False,
        ne_finetune=5,
):
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    n_plots = 4
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [[
            load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)),  # FedAvg + Full finetune
            load_pkl(get_ditto_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # Ditto
            load_pkl(get_pfedme_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # pFedMe
            load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, seed=seed, **args))  # PFL
        ] for seed in seeds]
    clean_name_lst = ["Finetune", "Ditto", "pFedMe", "Adapter"]
    _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0][0]
    
    for i in range(n_plots):
        test_deltas_lst = []
        for j, seed in enumerate(seeds):
            train_sizes, train_metrics, test_sizes, test_metrics = outs[j][i]
            # change in loss/accuracy
            if train:
                test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(train_metrics, train_metrics_fedavg)])
            else:
                test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[0] for t, t1 in zip(test_metrics, test_metrics_fedavg)])
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
        overall_mean = np.average(mean_per_client, weights=test_sizes)
        ax[i].plot(xs, mean_per_client, linestyle='dashed')
        ax[i].fill_between(xs, max_per_client, min_per_client, alpha=0.6)
        ax[i].set_xlabel('Client Id (Permuted)')
        if i==0: ax[i].set_ylabel(r'$\Delta$ ' + metric_short_rename_dict[metric_name])
        ax[i].set_title(clean_name_lst[i])
        ax[i].axhline(y=0, alpha=0.3, **zero_params)
        ax[i].axhline(y=overall_mean, alpha=0.5, **mean_params)
    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists


def per_user_stats_scatter_plot_full_v_partial_5(ds_and_model, seeds=list(range(1, 6)), pfl_algo="pfl_alternating", 
        args={}, metric_name='accuracy', min_is_best=False, 
        ne_finetune=5,
):
# Generalization gap per client
    style = {'color': COLORS[0], 'marker': 'o', 's':100, 'linestyle':"dashed"}
    n_plots = 4
    f, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4), sharey=True)
    suptitle = f.suptitle(dataset_rename_dict[ds_and_model], fontsize=18)

    # Gather filenames
    name_lst = get_name_list(ds_and_model) # PFL names
    pfl_model_name = "adapter" if "adapter" in name_lst else "adapter_16"
    outs = [[
            load_pkl(get_fedavg_finetune_all_fn(ds_and_model, "finetune", seed=seed, num_epochs_finetune=ne_finetune)),  # FedAvg + Full finetune
            load_pkl(get_ditto_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # Ditto
            load_pkl(get_pfedme_finetune_all_fn(ds_and_model, seed=seed, num_epochs_finetune=ne_finetune)), # pFedMe
            load_pkl(get_pfl_finetune_all_fn(ds_and_model, pfl_model_name, pfl_algo, seed=seed, **args))  # PFL
        ] for seed in seeds]
    clean_name_lst = ["Finetune", "Ditto", "pFedMe", "Adapter"]
    # _, train_metrics_fedavg, _, test_metrics_fedavg = outs[0][0]
    
    for i in range(n_plots):
        test_deltas_lst = []
        for j, seed in enumerate(seeds):
            train_sizes, train_metrics, test_sizes, test_metrics = outs[j][i]
            # change in loss/accuracy
            test_deltas = np.asarray([t[metric_name].iloc[-1] - t1[metric_name].iloc[-1] for t, t1 in zip(train_metrics, test_metrics)])
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
        overall_mean = np.average(mean_per_client, weights=test_sizes)
        ax[i].plot(xs, mean_per_client, linestyle='dashed')
        ax[i].fill_between(xs, max_per_client, min_per_client, alpha=0.6)
        ax[i].set_xlabel('Client Id (Permuted)')
        if i==0: ax[i].set_ylabel(r'$\Delta$ ' + metric_short_rename_dict[metric_name])
        ax[i].set_title(clean_name_lst[i])
        ax[i].axhline(y=0, alpha=0.3, **zero_params)
        ax[i].axhline(y=overall_mean, alpha=0.5, **mean_params)
    plt.tight_layout()
    extra_artists = (suptitle,)
    return f, extra_artists

