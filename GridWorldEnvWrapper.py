from NeuralRewardMachines.RL.Env.Environment import GridWorldEnv 
from NeuralRewardMachines.RL.Env.FiniteStateMachine import DFA
import numpy

class GridWorldEnvWrapper(GridWorldEnv):
    def __init__(self, formula, render_mode="human", state_type="symbolic", use_dfa_state: bool = True, external_automaton: bool = False, external_automaton_formula: str = None, recompute_image_pkl: bool = False):
        super().__init__(formula=formula, render_mode=render_mode, state_type=state_type, use_dfa_state=use_dfa_state, recompute_image_pkl=recompute_image_pkl)
        if external_automaton:
            self.external_automaton = DFA(
                external_automaton_formula, self.formula[1], self.formula[2], dictionary_symbols=self.dictionary_symbols)
            self.external_automaton_state = 0
        else:
            self.external_automaton = None

    def reset(self):
        observation, reward, info = super().reset()
        if self.external_automaton:
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
        if self.external_automaton:
            # Update external automaton state
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
        return observation, reward, done, truncated, info