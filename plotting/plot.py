import os

import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
# from rliable.plot_utils import plot_sample_efficiency_curve
from utils import plot_sample_efficiency_curve

from utils import get_data

if __name__ == "__main__":

    ### PROPS ##################################################################################################
    from itertools import product
    from collections import namedtuple

    env_ids = [
        'Swimmer-v4',
        'Hopper-v4',
        'HalfCheetah-v4',
        'Walker2d-v4',
        'Ant-v4',
        'Humanoid-v4'
    ]

    seaborn.set_theme(style='whitegrid')

    n_rows = 2
    n_cols = 3
    fig = plt.figure(figsize=(n_cols*4,n_rows*4))
    i = 1

    for env_id in env_ids:
        ax = plt.subplot(n_rows, n_cols, i)
        i+=1

        x_dict = {}
        y_dict = {}

        for algo in ['ddpg', 'td3']:
            key = f"{algo}"
            results_dir = f"../results/{env_id}/{algo}"

            x, y = get_data(results_dir, x_name='timestep', y_name='return', filename='evaluations.npz')
            if y is not None:
                x_dict[key] = x
                y_dict[key] = y
        #
        # param_sweep = {
        #     'b': [1],
        #     'plr': [1e-3, 1e-4],
        #     'pe': [16],
        #     'pmb': [16],
        #     'pns': [64, 128, 256],
        #     'pc': [0.3],
        #     'pkl': [0.01, 0.03],
        # }
        # param_tuple = namedtuple('param_tuple', field_names=param_sweep.keys())
        # for params in product(*param_sweep.values()):
        #     subdir = ''
        #     for key, val in zip(param_sweep.keys(), params):
        #         subdir += f'/{key}_{val}'
        #     p = param_tuple(*params)
        #
        #     sampling_algo = 'props'
        #
        #     key = subdir
        #     results_dir = f"../chtc/results_3/props_s_ho_w/results/{env_id}/ppo/{sampling_algo}/{subdir}"
        #     x, y = get_data(results_dir, x_name='timestep', y_name='return', filename='evaluations.npz')
        #     if y is not None:
        #         x_dict[key] = x
        #         y_dict[key] = y

        results_dict = {algorithm: score for algorithm, score in y_dict.items()}
        aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
        scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=100)

        plot_sample_efficiency_curve(
            frames=x_dict,
            point_estimates=scores,
            interval_estimates=cis,
            ax=ax,
            algorithms=None,
            xlabel='Timestep',
            ylabel=f'Return',
            labelsize='large',
            ticklabelsize='large',
            # marker='',
        )
        plt.title(f'{env_id}', fontsize='large')

        # Use scientific notation for x-axis
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # set fontsize of scientific notation label
        ax.xaxis.get_offset_text().set_fontsize('large')

        plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.88)

    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize='large')

    save_dir = f'figures'
    save_name = f'return_sweep.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
