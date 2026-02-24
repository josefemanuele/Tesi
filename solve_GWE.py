''' Solve GridWorld environment. Implement different reinforcement learning strategies. 
    Perform learning and plot learning curves. 
'''
import argparse
from sympy import re
from NeuralRewardMachines.LTL_tasks import formulas, ltls
from GridWorldEnvWrapper import GridWorldEnvWrapper
from utils.DirectoryManager import DirectoryManager
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
import numpy as np
import torch
import ReinforcementLearning.PPO as PPO
import ReinforcementLearning.DDQN as DDQN
from datetime import datetime
import LTL
from utils.utils import unique_ordered_list

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on GridWorld environment.")
    parser.add_argument("--algorithm", type=str, default="PPO", help="Reinforcement learning algorithm to use")
    parser.add_argument("--formulas", type=int, default=10, help="Number of formulas to consider from LTL_tasks")
    parser.add_argument("--external-automaton", action='store_true', help="If set, use external automaton")
    parser.add_argument("--no-automaton", action='store_true', help="If set, do not use automaton states")
    parser.add_argument("--image-state", action='store_true', help="Set state type to image")
    parser.add_argument("--render-mode", type=str, default="human", help="Set render mode")
    parser.add_argument("--runs", type=int, default=3, help="Experiment runs per formula")
    parser.add_argument("--episodes", type=int, default=3_000, help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps per rollout")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for algorithm updates")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per rollout")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for optimizer")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size for the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-translations", action='store_true', help="Test translated LTL formulas")
    parser.add_argument("--recompute-image-pkl", action='store_true', help="If set, recompute and overwrite image pkl files")
    parser.add_argument("--check-markovianity", action='store_true', help="If set, check Markovianity of the environment")
    parser.add_argument("--add-baseline", action='store_true', help="If set, add baseline to the experiment")
    parser.add_argument("--test-partial-formulas", action='store_true', help="If set, test partial LTL formulas")
    parser.add_argument("--one", action='store_true', help="If set, use cuda device:1")
    args = parser.parse_args()
    dm = DirectoryManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.one:
        device = torch.device("cuda:1")
    print(f"Using device: {device}")
    use_automaton = not args.no_automaton
    external_automaton = args.external_automaton
    if external_automaton:
        use_automaton = True
    state_type = "image" if args.image_state else "symbolic"
    if args.test_translations:
        external_automaton = True
    if args.test_partial_formulas:
        external_automaton = True
    if args.check_markovianity:
        external_automaton = True
    start_time = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
    # Log experiment parameters
    with open(dm.get_experiment_folder() + "experiment_parameters.txt", "w") as f:
        f.write(f"# Experiment parameters:\n")
        f.write(f'# Device: {device}\n')
        f.write(f"# Algorithm: {args.algorithm}\n")
        f.write(f"# State type: {state_type}\n")
        f.write(f"# Recompute image pkl: {args.recompute_image_pkl}\n")
        f.write(f"# Render mode: {args.render_mode}\n")
        f.write(f"# Use automaton states: {use_automaton}\n")
        f.write(f"# Use external automaton: {external_automaton}\n")
        f.write(f"# Number of formulas: {args.formulas}\n")
        f.write(f"# Runs per formula: {args.runs}\n")
        f.write(f"# Training episodes: {args.episodes}, steps per rollout: {args.steps}, batch size: {args.batch}\n")
        f.write(f"# epochs: {args.epochs}, clip_epsilon: {args.clip_epsilon}, lr: {args.lr}\n") 
        f.write(f"# vf_coef: {args.vf_coef}, ent_coef: {args.ent_coef}, max_grad_norm: {args.max_grad_norm}\n")
        f.write(f"# hidden layer size: {args.hidden}\n")
        f.write(f"# seed: {args.seed}\n")
        f.write(f"# test_translations: {args.test_translations}\n")
        f.write(f"# check_markovianity: {args.check_markovianity}\n")
        f.write(f"# add_baseline: {args.add_baseline}\n")
        f.write(f"# test_partial_formulas: {args.test_partial_formulas}\n")
        f.write(f"# start_time: {start_time}\n")
    # TODO: Check seeding working properly
    set_seed(args.seed)
    # Load LTL formulas
    LTL.load_data()
    data = LTL.get_data()
    for d in data[:args.formulas]:
        formula = d['formula']
        n_symbols = d['num_symbols']
        description = d['description']
        name = d['name']
        external_automaton_formula = formula
        dm.set_formula_name(description.replace(" ", "_"))
        print(f"Running experiments for: {description}")
        # Dataframe collecting data from all runs
        dfs = list()
        ltls = {"upper_bound": formula}
        if args.test_translations:
            translated_ltls = d.get("filtered_symbolic_lang2ltl_translations", [])
            ltls.update({f"translation_{n}": translated_ltl for n, translated_ltl in enumerate(translated_ltls)})
            print(f"Testing translated LTL formulas: {list(enumerate(ltls))}")
        if args.test_partial_formulas:
            partial_formulas = d.get("partial_formulas", [])
            ltls.update({f"partial_{n}": partial_formula for n, partial_formula in enumerate(partial_formulas)})
        if args.add_baseline:
            ltls["baseline"] = 0
        print(f'Testing LTL formulas: {ltls.items()}')
        for ltl_tag, ltl in ltls.items():
            print(f"Testing LTL formula: {ltl} with tag: {ltl_tag}")
            _use_automaton = False if ltl_tag == "baseline" else use_automaton
            _external_automaton = False if ltl_tag == "baseline" else external_automaton
            _check_markovianity = False if ltl_tag == "baseline" else args.check_markovianity
            # Create environment
            env = GridWorldEnvWrapper(
                formula=(formula, n_symbols, name + '_' + ltl_tag), 
                render_mode=args.render_mode, 
                state_type=state_type, 
                use_dfa_state=_use_automaton, 
                external_automaton=_external_automaton, 
                external_automaton_formula=ltl,
                recompute_image_pkl=args.recompute_image_pkl,
                check_markovianity=_check_markovianity
                )
            # List of external formulas / list of automatas / for loop iterating over code.
            runs = list()
            for r in range(1, args.runs + 1):
                print(f"Experiment {r} / {args.runs}")
                if args.algorithm == "DDQN":
                    _, data = DDQN.train_ddqn(device=device, env=env, hidden=args.hidden, episodes=args.episodes, max_steps=args.steps, batch_size=args.batch)
                elif args.algorithm == "PPO":
                    _, data = PPO.train_ppo(device=device, env=env, hidden=args.hidden, episodes=args.episodes, steps=args.steps, minibatch_size=args.batch, 
                            epochs=args.epochs, clip_epsilon=args.clip_epsilon, lr=args.lr, vf_coef=args.vf_coef, 
                            ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm)
                else:
                    raise ValueError(f"ERROR: Algorithm {args.algorithm} not supported.")
                # Add run column to data
                df = pd.DataFrame(data, columns=["episode", "episode_reward", "episode_length", "done", "truncated", "total_steps"])
                df["reward_rolling_avg"] = df["episode_reward"].rolling(window=100, min_periods=1).mean()
                df["run"] = r
                df["ltl_tag"] = ltl_tag
                df["ltl"] = ltl
                if _check_markovianity:
                    markovianity_statistics = env.get_markovianity_statistics()
                    with open(dm.get_formula_folder() + f"markovianity_ltl_{ltl_tag}_r_{r}.txt", "w") as f:
                        f.write(f"Markovianity statistics for LTL formula: {ltl}\n")
                        f.write(f"Correct formula: {formula}\n")
                        f.write(f"Is Markovian: {env.is_markovian()}\n")
                        f.write(f"Number of states: {env.get_num_states()}\n")
                        f.write(f"Number of observed transitions: {len(env.get_markovianity())}\n")
                        f.write(f"Global Markovianity Rate: {env.get_global_markovianity_rate()}\n")
                        f.write(f"Transition rewards mapping:\n")
                        for transition, rewards_info in markovianity_statistics.items():
                            for reward, info in rewards_info.items():
                                f.write(f"Transition {transition}, Reward {reward}, Count {info['count']}, Percentage {info['percentage']:.2f}\n")
                    df["markovianity_rate"] = env.get_global_markovianity_rate()
                if args.add_baseline and ltl_tag == "baseline":
                    df["markovianity_rate"] = 0
                runs.append(df)
            ltl_df = pd.concat(runs, ignore_index=True)
            if args.check_markovianity:
                ltl_df['avg_markovianity'] = ltl_df['markovianity_rate'].mean()
            dfs.append(ltl_df)
        # Concatenate dataframes from all runs
        dataframe = pd.concat(dfs, ignore_index=True)
        # Plot learning curves
        print(f"Plotting learning curves for: {description}")
        differentiator = 'ltl_tag' # if args.test_translations else 'run'
        aggregated_differentiator = 'ltl_tag' #if args.test_translations else None
        # palette = sns.color_palette("rocket_r", as_cmap=True) 
        hue = None
        hue_norm = None
        if args.check_markovianity:
            hue = 'avg_markovianity'
            min_index = 1 if args.add_baseline else 0
            markovianity_values = dataframe[hue].dropna().unique()
            markovianity_values.sort()
            min_markovianity = markovianity_values[min_index] 
            max_markovianity = markovianity_values[-1]
            hue_norm = ((max_markovianity - ((max_markovianity - min_markovianity) * 2)), max_markovianity) if len(markovianity_values) > 2 else None
            print(f"Markovianity values: {markovianity_values}, min_markovianity: {min_markovianity}, max_markovianity: {max_markovianity}")
            print(f'Hue: {hue}, Hue norm: {hue_norm}')
        # hue_norm = None if args.test_partial_formulas else hue_norm
        # plt.title(f"{args.algorithm} Learning Curve per {differentiator} - {description}")
        # Plot differentated learning curves
        sns.relplot(data=dataframe, kind="line", x="episode", y="reward_rolling_avg", 
                    style=differentiator, col=differentiator, hue=hue, hue_norm=hue_norm)
        plt.savefig(dm.get_formula_folder() + f"{args.algorithm}_Learning_Curve_per_{differentiator}.png")
        plt.close()
        # plt.title(f"{args.algorithm} Learning Curve - {description}")
        # Plot aggregated learning curve
        sns.relplot(data=dataframe, kind="line", x="episode", y="reward_rolling_avg", 
                    style=aggregated_differentiator, hue=hue, hue_norm=hue_norm)
        plt.savefig(dm.get_formula_folder() + f"{args.algorithm}_Learning_Curve.png")
        plt.close()
        # Save dataframe to CSV
        dataframe.to_csv(dm.get_formula_folder() + f"{args.algorithm}_Training_data.csv", index=False)
    # Log end time
    end_time = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
    with open(dm.get_experiment_folder() + "experiment_parameters.txt", "a") as f:
        f.write(f"# end_time: {end_time}\n")
    print(f"Saved data into: {dm.get_experiment_folder()}")