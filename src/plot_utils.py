
import matplotlib.pyplot as plt
import numpy as np
#!pip3 install pickle5
import seaborn as sb
import pickle5 as pickle

def plot_and_save_master_episodes(master_episodes, file_path):
    max_numbers = [k for k in master_episodes]
    episodes = [v for v in master_episodes.values()]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(max_numbers, episodes, s=100, color='black')
    ax.set_xticks(max_numbers)
    ax.set_ylim(0, episodes[-1] + episodes[-1] // 10)
    ax.set_xlabel('Maximum number presented', fontsize=26)
    ax.set_ylabel('Master Episode', fontsize=26)
    ax.tick_params(labelsize=20)
    fig.savefig(file_path)


# If ext_tool==Abacus: calculate spatial structure:
# position of token correlate with the presented number
def plot_and_save_token_positions_of_reprs(ext_repr, file_path):
    one_repr_for_each_number = [repr[0] for repr in ext_repr.values()]
    numbers = [k for k, v in ext_repr.items()]
    repr_size = ext_repr[0][0].size
    pos_of_token_for_each_number = [np.where(repry == 1)[0][0] for repry in one_repr_for_each_number]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(numbers, pos_of_token_for_each_number, s=100, color='black')
    ax.set_xticks(numbers)
    ax.set_yticks(range(repr_size))
    # ax.set_ylim(0, episodes[-1]+episodes[-1]//10)
    ax.set_xlabel('Presented number', fontsize=26)
    ax.set_ylabel('Position of token', fontsize=26)
    ax.tick_params(labelsize=20)
    fig.savefig(file_path)


def plot_and_save_external_repr_correlation(ext_repr, task_description, file_path):
    numbers = list(ext_repr.keys())
    max_objects = numbers[-1]
    first, second = max_objects, max_objects
    var_len = first
    range_var_len = range(var_len)
    combined = [(f, s) for f in numbers for s in numbers]

    corr_matrix = np.empty([var_len + 1, var_len + 1])
    for (i, j) in combined:
        vec1 = np.array(ext_repr[i][0])
        vec2 = np.array(ext_repr[j][0])
        vec3 = np.stack([vec1, vec2])[:, :, 0]
        corry = np.corrcoef(vec3)[0, 1]
        corr_matrix[i, j] = corry

    fig, ax = plt.subplots(figsize=(16, 14))
    title = task_description
    tick_labels = [str(i) for i in range(0, max_objects + 1)]  # later all keys in DICT of ext_representation
    if (ax is None):
        ax = plt.axes()
    sb.heatmap(corr_matrix,
               xticklabels=tick_labels,
               yticklabels=tick_labels,
               cmap='RdBu_r',
               linewidth=0.5,
               ax=ax,
               vmin=-0.2,
               vmax=1.0,
               annot=False)

    ax.set_title(title, fontsize=40)
    ax.set_xlabel('Presented number', fontsize=26)
    ax.set_ylabel('Presented number', fontsize=26)
    ax.tick_params(labelsize=20)
    # file_path = exp_dir + '/ext_repr_correlation.svg'
    fig.savefig(file_path)
