from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv 
from NeuralRewardMachines.RL.Env.FiniteStateMachine import DFA
import numpy

class GridWorldEnvWrapper(GridWorldEnv):
    def __init__(self, formula, render_mode="human", state_type="symbolic", use_dfa_state: bool = True, 
                 external_automaton: bool = False, external_automaton_formula: str = None, recompute_image_pkl: bool = False, 
                 check_markovianity: bool = False):
        super().__init__(formula=formula, render_mode=render_mode, state_type=state_type, 
                         use_dfa_state=use_dfa_state, recompute_image_pkl=recompute_image_pkl)
        if check_markovianity:
            self.markovianity = {}  # For checking Markovianity
            external_automaton = True
        if external_automaton:
            self.external_automaton = DFA(
                external_automaton_formula, self.formula[1], self.formula[2] + external_automaton_formula + "_external", dictionary_symbols=self.dictionary_symbols)
            self.external_automaton_state = 0

    def update_markovianity(self, state, symbol, reward):
        transition = (state, symbol)
        rewards_count = self.markovianity.get(transition, dict())
        rewards_count[reward] = rewards_count.get(reward, 0) + 1
        self.markovianity[transition] = rewards_count

    def is_markovian(self):
        for transition, rewards in self.markovianity.items():
            if len(rewards) > 1:
                return False
        return True
    
    def get_markovianity_statistics(self):
        self.markovianity_stats = {}
        for transition, rewards_count in self.markovianity.items():
            for reward, count in rewards_count.items():
                self.markovianity_stats[transition] = self.markovianity_stats.get(transition, {})
                self.markovianity_stats[transition][reward] = {'count': count, 'percentage': count / sum(rewards_count.values())}
        return self.markovianity_stats
    
    def get_global_markovianity_rate(self):
        # global_markovianity_rate = sum([max([reward['percentage'] for reward in self.markovianity_stats[transition].values()]) for transition in self.markovianity]) / len(self.markovianity)
        sum_ = sum([max([reward['percentage'] for reward in self.markovianity_stats[transition].values()]) for transition in self.markovianity])
        transitions = len(self.markovianity)
        global_markovianity_rate = f'{sum_} / {transitions} = {sum_ / transitions}'
        return global_markovianity_rate
    
    def get_markovianity(self):
        return self.markovianity
    
    def get_num_states(self):
        if hasattr(self, 'external_automaton'):
            return self.external_automaton.num_of_states
        else:
            return None

    def reset(self):
        observation, reward, info = super().reset()
        if hasattr(self, 'external_automaton'):
            self.external_automaton_state = 0
            # Update observation
            if self.state_type == 'symbolic':
                # Replace automaton state in observation
                observation = numpy.append(observation[:-1], [self.external_automaton_state])
            elif self.state_type == 'image':
                one_hot_dfa_state = [0 for _ in range(self.external_automaton.num_of_states)]
                one_hot_dfa_state[self.external_automaton_state] = 1
                # Replace automaton state in observation tuple
                observation = [one_hot_dfa_state, observation[1]]
        return observation, reward, info

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        if hasattr(self, 'external_automaton'):
            # Update external automaton state
            previous_external_automaton_state = self.external_automaton_state
            sym = super()._current_symbol()
            self.external_automaton_state = self.external_automaton.transitions[self.external_automaton_state][sym]
            # Update observation
            if self.state_type == 'symbolic':
                # Replace automaton state in observation
                observation = numpy.append(observation[:-1], [self.external_automaton_state])
            elif self.state_type == 'image':
                one_hot_dfa_state = [0 for _ in range(self.external_automaton.num_of_states)]
                one_hot_dfa_state[self.external_automaton_state] = 1
                # Replace automaton state in observation tuple
                observation = [one_hot_dfa_state, observation[1]]
        if hasattr(self, 'markovianity'):
            self.update_markovianity(previous_external_automaton_state, sym, reward)
        return observation, reward, done, truncated, info