import gymnasium as gym
from stable_baselines3 import PPO
from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv
from NeuralRewardMachines.LTL_tasks import formulas

formula = formulas[0]
env = GridWorldEnv(formula=formula)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")