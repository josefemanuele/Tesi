''' Solve GridWorld environment. Implement different reinforcement learning strategies. 
    Perform learning and plot learning curves. 
'''
import argparse
from sympy import re
from NeuralRewardMachines.LTL_tasks import formulas, ltls
from GridWorldEnvWrapper import GridWorldEnvWrapper
from utils.DirectoryManager import DirectoryManager
import pandas
from matplotlib import pyplot as plt
import seaborn
import random
import numpy as np
import torch
import ReinforcementLearning.PPO as PPO
import ReinforcementLearning.DDQN as DDQN
from datetime import datetime
import LTL
from utils.utils import unique_ordered_list

dm = DirectoryManager()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    parser.add_argument("--runs", type=int, default=8, help="Experiment runs per formula")
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps per rollout")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Minibatch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per rollout")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for optimizer")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size for the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-translations", action='store_true', help="Test translated LTL formulas")
    parser.add_argument("--recompute-image-pkl", action='store_true', help="If set, recompute and overwrite image pkl files")
    parser.add_argument("--check-markovianity", action='store_true', help="If set, check Markovianity of the environment")
    args = parser.parse_args()
    use_automaton = not args.no_automaton
    external_automaton = args.external_automaton
    if external_automaton:
        use_automaton = True
    state_type = "image" if args.image_state else "symbolic"
    runs = args.runs
    if args.test_translations:
        runs = 1
        external_automaton = True
    if args.check_markovianity:
        runs = 1
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
        f.write(f"# Runs per formula: {runs}\n")
        f.write(f"# Training episodes: {args.episodes}, steps per rollout: {args.steps}, minibatch size: {args.minibatch_size}\n")
        f.write(f"# epochs: {args.epochs}, clip_epsilon: {args.clip_epsilon}, lr: {args.lr}\n") 
        f.write(f"# vf_coef: {args.vf_coef}, ent_coef: {args.ent_coef}, max_grad_norm: {args.max_grad_norm}\n")
        f.write(f"# hidden layer size: {args.hidden}\n")
        f.write(f"# seed: {args.seed}\n")
        f.write(f"# test_translations: {args.test_translations}\n")
        f.write(f"# check_markovianity: {args.check_markovianity}\n")
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
        ltls = [formula]
        if args.test_translations:
            translated_ltls = d.get("filtered_symbolic_lang2ltl_translations", [])
            ltls.extend(unique_ordered_list(translated_ltls))
            print(f"Testing translated LTL formulas: {list(enumerate(ltls))}")
        for n_ltl, ltl in enumerate(ltls):
            # Create environment
            env = GridWorldEnvWrapper(
                formula=(formula, n_symbols, name + '_' + str(n_ltl)), 
                render_mode=args.render_mode, 
                state_type=state_type, 
                use_dfa_state=use_automaton, 
                external_automaton=external_automaton, 
                external_automaton_formula=ltl,
                recompute_image_pkl=args.recompute_image_pkl,
                check_markovianity=args.check_markovianity
                )
            # List of external formulas / list of automatas / for loop iterating over code.
            for r in range(1, runs + 1):
                print(f"Experiment {r} / {runs}")
                if args.algorithm == "DDQN":
                    _, data = DDQN.train_ddqn(device=device, env=env, episodes=args.episodes, max_steps=args.steps)
                elif args.algorithm == "PPO":
                    _, data = PPO.train_ppo(device=device, env=env, hidden=args.hidden, episodes=args.episodes, steps=args.steps, minibatch_size=args.minibatch_size, 
                            epochs=args.epochs, clip_epsilon=args.clip_epsilon, lr=args.lr, vf_coef=args.vf_coef, 
                            ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm)
                else:
                    raise ValueError(f"ERROR: Algorithm {args.algorithm} not supported.")
                # Add run column to data
                df = pandas.DataFrame(data, columns=["episode", "episode_reward", "episode_length", "done", "truncated", "total_steps"])
                df["reward_rolling_avg"] = df["episode_reward"].rolling(window=100, min_periods=1).mean()
                df["run"] = r
                df["n_ltl"] = n_ltl
                df["ltl"] = ltl
                dfs.append(df)
                if args.check_markovianity:
                    markovianity_statistics = env.get_markovianity_statistics()
                    with open(dm.get_formula_folder() + f"markovianity_ltl_{n_ltl}.txt", "w") as f:
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
        # Concatenate dataframes from all runs
        dataframe = pandas.concat(dfs, ignore_index=True)
        # Plot learning curves
        differentiator = 'n_ltl' if args.test_translations else 'run'
        print(f"Plotting learning curves for: {description}")
        plt.title(f"{args.algorithm} Learning Curve - {description}")
        seaborn.relplot(data=dataframe, kind="line", x="episode", y="reward_rolling_avg", hue=differentiator)
        plt.savefig(dm.get_formula_folder() + f"{args.algorithm}_Learning_Curve.png")
        plt.close()
        plt.title(f"{args.algorithm} Learning Curve per run - {description}")
        seaborn.relplot(data=dataframe, kind="line", x="episode", y="reward_rolling_avg", col=differentiator, hue=differentiator)
        plt.savefig(dm.get_formula_folder() + f"{args.algorithm}_Learning_Curve_per_run.png")
        plt.close()
        # Save dataframe to CSV
        dataframe.to_csv(dm.get_formula_folder() + f"{args.algorithm}_Training_data.csv", index=False)
    # Log end time
    end_time = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
    with open(dm.get_experiment_folder() + "experiment_parameters.txt", "a") as f:
        f.write(f"# end_time: {end_time}\n")