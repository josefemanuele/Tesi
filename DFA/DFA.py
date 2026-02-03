
from flloat.parser.ltlf import LTLfParser
# TODO: Use ltl2dfa
# from ltlf2dfa.parser.ltlf import LTLfParser

class DFA:
    def __init__(self, ltl_formula, num_symbols, formula_name, dictionary_symbols):
        self.dictionary_symbols = dictionary_symbols
        #From LTL to DFA
        parser = LTLfParser()
        ltl_formula_parsed = parser(ltl_formula)
        dfa = ltl_formula_parsed.to_automaton()
        # dfa = ltl_formula_parsed.to_dfa()
        # print("Formula:", ltl_formula_parsed)
        # print("dfa:", dfa)
        # print the automaton
        graph = dfa.to_graphviz()
        graph.render("data/symbolicDFAs/"+formula_name)
        #From symbolic DFA to simple DFA
        # print("DFA dict:", dfa.__dict__)
        self.alphabet = ["c" + str(i) for i in range(num_symbols)]
        self.transitions = self.reduce_dfa(dfa)
        print("Transitions: ", self.transitions)
        self.num_of_states = len(self.transitions)
        self.acceptance = []
        for s in range(self.num_of_states):
            if s in dfa._final_states:
                self.acceptance.append(True)
            else:
                self.acceptance.append(False)

    # Reduce the DFA obtained from pythomata library to a simple DFA with integer-labeled transitions
    def reduce_dfa(self, pythomata_dfa):
        dfa = pythomata_dfa
        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i, sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)
        red_trans_funct = {}
        for s0 in dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key
        return red_trans_funct