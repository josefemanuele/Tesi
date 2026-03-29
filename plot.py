## Load the data files.
## Merge the data files.
## Replot the data.
## Save the plots.
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils.DirectoryManager import DirectoryManager
import argparse
import os
import json
import pathlib

task_names = ['task1:_visit(pickaxe,_lava)', 
                'task2:_visit(pickaxe,_lava,_door)',
                'task3:_seq_visit(pickaxe,_lava)',
                'task4:_seq_visit(pickaxe,_lava)_+_seq_visit(door,_gem)',
                'task5:_seq_visit(pickaxe,_lava)_+_visit(door)',
                'task6:_seq_visit(pickaxe,_lava)_+_visit(door,_gem)',
                'task7:_visit(pickaxe,_lava)_+_glob_av(door)',
                'task8:_visit(pickaxe,_lava)_+_glob_av(door)_+_glob_av(gem)',
                'task9:_seq_visit(pickaxe,_lava)_+_glob_av(door)',
                'task10:_seq_visit(pickaxe,_lava)_+_glob_av(door)_+_glob_av(gem)']

max_episode = 3_000

dataset_fetch = ['symbolic_lang2ltl_translations', 'poll_translations_to_GPT5']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot datasets.")
    parser.add_argument("algorithm", type=str, help="Algorithm used for training.")
    parser.add_argument("--translation_engine", type=str, nargs='+', help="Algorithm used for training.")
    parser.add_argument("--max_episode", type=int, default=3_000, help="Experiment to load data.")
    parser.add_argument("--experiment", type=str, nargs='+', help="Experiment to load data.")
    parser.add_argument("--baseline", type=str, help="Baseline execution data.")
    parser.add_argument("--upper_bound", type=str, help="Upper bound execution data.")
    parser.add_argument("--rnn", type=str, help="RNN execution data.")
    args = parser.parse_args()
    print(args.upper_bound, args.rnn, args.baseline, args.experiment, args.translation_engine)
    with open('data/Lang2LTL/LTLs.json') as f:
        ltls_dataset = json.load(f)
    # Load experiments
    if args.upper_bound:
        upper_bound_dm = DirectoryManager(args.upper_bound)
        upper_bound_experiment_folder = upper_bound_dm.get_experiment_folder()
    if args.rnn:
        rnn_dm = DirectoryManager(args.rnn)
        rnn_experiment_folder = rnn_dm.get_experiment_folder()
    if args.baseline:
        baseline_dm = DirectoryManager(args.baseline)
        baseline_experiment_folder = baseline_dm.get_experiment_folder()
    experiment_folders = []
    for i, e in enumerate(args.experiment):
        e_dm = DirectoryManager(e)
        experiment_folder = e_dm.get_experiment_folder()
        experiment_folders.append(experiment_folder)
    # Load data
    df_list = []
    plot_dm = DirectoryManager()
    for n, t in enumerate(task_names):
        if args.upper_bound:
            # Load upper bound
            upper_bound_data = os.path.join(upper_bound_experiment_folder, t, args.algorithm + "_Training_data.csv")
            ub_df = pd.read_csv(upper_bound_data)
            # Drop episodes over max_episode
            ub_df = ub_df[ub_df['episode'] <= args.max_episode]
            # Get upper bound
            # g = df[df['n_ltl'] == 0]
            g = ub_df[ub_df['ltl_tag'] == 'upper_bound']
            f = g.groupby(['episode'], as_index=False)['reward_rolling_avg'].mean().rename(
                    columns={'reward_rolling_avg':'m_average'})
            f['m_tag'] = 'upper_bound'
            df_list.append(f)
        if args.rnn:
            # Load rnn
            rnn_data = os.path.join(rnn_experiment_folder, t, args.algorithm + "_Training_data.csv")
            rnn_df = pd.read_csv(rnn_data)
            # Drop episodes over max_episode
            rnn_df = rnn_df[rnn_df['episode'] <= args.max_episode]
            # Get rnn
            g = rnn_df[rnn_df['ltl_tag'] == 'rnn']
            f = g.groupby(['episode'], as_index=False)['reward_rolling_avg'].mean().rename(
                    columns={'reward_rolling_avg':'m_average'})
            f['m_tag'] = 'rnn'
            df_list.append(f)
        if args.baseline:
            # Load baseline
            baseline_data = os.path.join(baseline_experiment_folder, t, args.algorithm + "_Training_data.csv")
            baseline_df = pd.read_csv(baseline_data)
            # Drop episodes over max_episode
            baseline_df = baseline_df[baseline_df['episode'] <= args.max_episode]
            # Get baseline
            g = baseline_df[baseline_df['ltl_tag'] == 'baseline']
            f = g.groupby(['episode'], as_index=False)['reward_rolling_avg'].mean().rename(
                    columns={'reward_rolling_avg':'m_average'})
            f['m_tag'] = 'baseline'
            df_list.append(f)
        for i, experiment_folder in enumerate(experiment_folders):
            experiment_data = os.path.join(experiment_folder, t, args.algorithm + "_Training_data.csv")
            e_df = pd.read_csv(experiment_data)
            # Drop episodes over max_episode
            e_df = e_df[e_df['episode'] <= args.max_episode]
            # Get experiments mean
            # g = df.groupby(['n_ltl', 'ltl', 'episode'], 
            g = e_df.groupby(['ltl', 'episode'], 
                as_index=False)['reward_rolling_avg'].mean().rename(
                    columns={'reward_rolling_avg':'m_average'})
            # Apply weight
            first = True
            first_ltl = None
            other_weights = 0.0
            for ltl in g['ltl'].unique():
                if first:
                    first_ltl = ltl
                    first = False
                    continue
                counts = ltls_dataset[n][dataset_fetch[i]].count(ltl)
                weight = counts / 10
                other_weights = other_weights + weight
                print(f'{t} ltl: {ltl} : {weight:.2f}')
                mask = g['ltl'] == ltl
                g.loc[mask, 'm_average'] = g.loc[mask, 'm_average'] * weight
            first_weight = 1 - other_weights
            print(f'{t} first ltl: {first_ltl} : {first_weight:.2f}')
            mask = g['ltl'] == first_ltl
            g.loc[mask, 'm_average'] = g.loc[mask, 'm_average'] * first_weight
            # Sum weights
            f = g.groupby(['episode'], as_index=False)['m_average'].sum()
            f['m_tag'] = args.translation_engine[i]
            df_list.append(f)
        dataset = pd.concat(df_list, ignore_index=True)
        # Plot
        plot_dm.set_formula_name(t)
        task_folder = plot_dm.get_formula_folder()
        plt.title(f"{args.algorithm} Weighted averaged runs - {t}")
        sns.relplot(data=dataset, kind="line", x="episode", y="m_average", hue='m_tag')
        plt.savefig(task_folder + f"{args.algorithm}_Learning_Curve_merged.png")
        plt.close()
        df_list = []
        print()