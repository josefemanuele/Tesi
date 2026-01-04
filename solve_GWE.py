''' Solve GridWorld environment. Implement different reinforcement learning strategies. 
    Perform learning and plot learning curves. 
'''
import argparse
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

dm = DirectoryManager()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supported_algorithms = ["DDQN", "PPO"]

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on GridWorld environment.")
    parser.add_argument("--algorithm", type=str, default="DDQN", help="Reinforcement learning algorithm to use")
    parser.add_argument("--formulas", type=int, default=10, help="Number of formulas to consider from LTL_tasks")
    parser.add_argument("--external-automaton", action='store_true', help="If set, use external automaton")
    parser.add_argument("--no-automaton", action='store_true', help="If set, do not use automaton states")
    parser.add_argument("--image-state", action='store_true', help="Set state type to image")
    parser.add_argument("--runs", type=int, default=3, help="Experiment runs per formula")
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
    args = parser.parse_args()
    algorithm = args.algorithm
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {supported_algorithms}")
    n_formulas = args.formulas
    use_automaton = not args.no_automaton
    external_automaton = args.external_automaton
    if external_automaton:
        use_automaton = True
    state_type = "image" if args.image_state else "symbolic"
    runs = args.runs
    episodes = args.episodes
    steps = args.steps
    minibatch_size = args.minibatch_size
    epochs = args.epochs
    clip_epsilon = args.clip_epsilon
    lr = args.lr
    vf_coef = args.vf_coef
    ent_coef = args.ent_coef
    max_grad_norm = args.max_grad_norm
    hidden = args.hidden
    seed = args.seed
    # Log experiment parameters
    with open(dm.get_experiment_folder() + "experiment_parameters.txt", "w") as f:
        f.write(f"# Experiment parameters:\n")
        f.write(f"# Number of formulas: {n_formulas}\n")
        f.write(f"# Use automaton states: {use_automaton}\n")
        f.write(f"# Use external automaton: {external_automaton}\n")
        f.write(f"# State type: {state_type}\n")
        f.write(f"# Runs per formula: {runs}\n")
        f.write(f"# Training episodes: {episodes}, steps per rollout: {steps}, minibatch size: {minibatch_size}\n")
        f.write(f"# epochs: {epochs}, clip_epsilon: {clip_epsilon}, lr: {lr}\n") 
        f.write(f"# vf_coef: {vf_coef}, ent_coef: {ent_coef}, max_grad_norm: {max_grad_norm}\n")
        f.write(f"# hidden layer size: {hidden}\n")
        f.write(f"# seed: {seed}\n")
    set_seed(seed)
    formulas = formulas[:n_formulas]
    for i in range(len(formulas)):
        formula = formulas[i]
        ltl = ltls[i]
        dm.set_formula_name(formula[2].replace(" ", "_"))
        print(f"Running experiments for: {formula[2]}")
        # Dataframe collecting data from all runs
        dataframe = pandas.DataFrame([], columns=["episode", "episode_reward", "episode_length", "done", "truncated", "total_steps", "avg_reward", "run"])
        # Create environment
        env = GridWorldEnvWrapper(formula=formula, state_type=state_type, use_dfa_state=use_automaton, external_automaton=external_automaton, ltl=ltl)
        for r in range(1, runs + 1):
            print(f"Experiment {r} / {runs}")
            if algorithm == "DDQN":
                _, data = DDQN.train_ddqn(device=device, env=env, episodes=episodes, max_steps=steps)
            if algorithm == "PPO":
                # Train PPO
                _, data = PPO.train_ppo(device=device, env=env, hidden=hidden, episodes=episodes, steps=steps, minibatch_size=minibatch_size, 
                        epochs=epochs, clip_epsilon=clip_epsilon, lr=lr, vf_coef=vf_coef, 
                        ent_coef=ent_coef, max_grad_norm=max_grad_norm)
            # Add run column to data
            df = pandas.DataFrame(data, columns=["episode", "episode_reward", "episode_length", "done", "truncated", "total_steps", "avg_reward"])
            df["run"] = r
            dataframe = pandas.concat([dataframe, df], ignore_index=True)
        # Plot learning curves
        print(f"Plotting learning curves for: {formula[2]}")
        plt.title(f"{algorithm} Learning Curve - {formula[2]}")
        seaborn.relplot(data=dataframe, kind="line", x="episode", y="avg_reward")
        plt.savefig(dm.get_plot_folder() + f"{algorithm}_Learning_Curve.png")
        plt.clf()
        plt.title(f"{algorithm} Learning Curve per run - {formula[2]}")
        seaborn.relplot(data=dataframe, kind="line", x="episode", y="avg_reward", col="run", hue="run")
        plt.savefig(dm.get_plot_folder() + f"{algorithm}_Learning_Curve_per_run.png")
        plt.clf()
        # TODO: Save dataframe to CSV (?)