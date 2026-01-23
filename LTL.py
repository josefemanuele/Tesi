import argparse
from openai import OpenAI
import time
from NeuralRewardMachines.LTL_tasks import formulas, utterances
from DFA.DFA import DFA
import json
from datetime import datetime
from Lang2LTLWrapper import translate

models = ['gpt-5.2', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano']
models = models[:]

dictionary_symbols = ['P', 'L', 'D', 'G', 'E' ]

data = []

data_path = 'data/Lang2LTL/LTLs.json'

# Utterances corresponding to the formulas.
utterances = []
utterances.append("Visit the pickaxe and visit the lava.")
utterances.append("Visit the pickaxe, the lava, and the door.")
utterances.append("First visit the pickaxe, then visit the lava.")
utterances.append("First visit the pickaxe, then visit the lava. After that, first visit the door, then visit the gem.")
utterances.append("First visit the pickaxe, then visit the lava. Also, visit the door.")
utterances.append("First visit the pickaxe, then visit the lava. Also, visit the door and the gem.")
utterances.append("Visit the pickaxe and the lava, always avoid the door.")
utterances.append("Visit the pickaxe and the lava, always avoid the door and the gem.")
utterances.append("First visit the pickaxe, then visit the lava. Always avoid the door.")
utterances.append("First visit the pickaxe, then visit the lava. Always avoid the door and the gem.")

def create_data():
    for i, formula in enumerate(formulas):
        data.append({"formula": formula[0], "num_symbols": formula[1], "description": formula[2], "utterance": utterances[i]})

def store_data():
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_data():
    with open(data_path, 'r') as f:
        global data
        data = json.load(f)

def get_data():
    return data

task = 'Transform the following natural language sentence into an Linear Temporal Logic task. ' \
'Use & as AND operator, | as OR, > as implication, ! as NOT. Use F as finally temporal operator, ' \
'G as globally temporal operator. Use c0 as symbol for pickaxe, c1 as symbol for lava, c2 as symbol ' \
'for door, c3 as symbol for gem. Only provide the LTL task as your response,  without the triple ' \
'backtick tag defining a code block. ' \

sentences = ['Pick the pickaxe and pick the gem.',
             'Pick the gem, only after picking the pickaxe.',
             'First pick the pickaxe, then pick the gem, and finally go to the door.',
             'Pick the pickaxe and the gem, but never touch the lava.',
             'First pick the axe, then take the diamond. Make sure to avoid the lava at all times.',
             'Go to the door only after picking up the pickaxe and the gem, while avoiding the lava.',
             'Go to the door as soon as possible, avoid the stone, pick up the pickaxe along the way.']

log = 'data/LTL_log.csv'
from pathlib import Path
if not Path(log).exists():
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        f.write('model;time;sentence;ltl_task\n')

def evaluate_models():
    for model in models:
        sentences = utterances
        for sentence in sentences:
            input = task + sentence
            start = time.time()
            # response = client.responses.create(
            #     prompt={
            #         "id": "pmpt_69430a7f80e08196afa1b5ca4ab0759501e931d0639b3d51",
            #         "version": "2",
            #         "variables": {
            #         "sentence": sentence,
            #         }
            #     }
            # )
            response = client.responses.create(
                model=model,
                input=input
            )
            end = time.time()
            response_time = end - start
            line = f'{model};{response_time:.3f};{sentence};{response.output_text}'
            print(line)
            with open(log, 'a') as f:
                f.write(line + '\n')

def data_augmentation(client, model='gpt-5.2', utterance="Visit the pickaxe and visit the lava."):
    prompt = "You are given a natural language instruction for a task in a simulated environment. " \
    "Your goal is to generate ten different paraphrased versions of the instruction, " \
    "while preserving the original meaning. " \
    "Avoid using the same verbs in the same sentences." \
    "Paraphrase the naming of the objects too. " \
    "Provide the paraphrased instructions one after the other, separated by line breaks. " \
    "Without any additional numbering or bullet points. " \
    "In only ascii characters."
    input = prompt + "\n\nOriginal instruction: " + utterance
    response = client.responses.create(
        model=model,
        input=input
    )
    paraphrased = response.output_text.split('\n')
    return paraphrased

def lang_to_ltl(client, model='gpt-5.2', utterance="Visit the pickaxe and visit the lava."):
    prompt = "You are given a natural language instruction for a task in a simulated environment. " \
    "Your goal is to convert the instruction into a Linear Temporal Logic (LTL) formula. " \
    "Use & as AND operator, | as OR, > as implication, ! as NOT. Use F as finally temporal operator, " \
    "G as globally temporal operator. Use c0 as symbol for pickaxe, c1 as symbol for lava, c2 as symbol " \
    "for door, c3 as symbol for gem. Only provide the LTL formula as your response, without the triple " \
    "backtick tag defining a code block. "
    input = prompt + "\n\nOriginal instruction: " + utterance
    response = client.responses.create(
        model=model,
        input=input
    )
    ltl = response.output_text.strip()
    return ltl

def check_equivalence(ltl1, ltl2, num_symbols):
    # equivalence = spot.are_equivalent(ltl1, ltl2)
    # equivalence_formula = f"(({ltl1}) -> ({ltl2})) & (({ltl2}) -> ({ltl1}))"
    equivalence_formula = f"({ltl1}) <-> ({ltl2})"
    equivalence_automaton = DFA(ltl_formula=equivalence_formula,
                                        num_symbols=num_symbols, 
                                        formula_name="equivalence_check", 
                                        dictionary_symbols=dictionary_symbols)
    # equivalence_automaton.write_dot_file(f"data/equivalence_check_{i}.dot")
    equivalence = (equivalence_automaton.num_of_states == 1 and equivalence_automaton.acceptance[0] == True)
    return equivalence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle LTL tasks with LLMs")
    parser.add_argument("--model", type=str, default='gpt-5.2', help="Language model to use")
    parser.add_argument("--create-data", default=False, help="Whether to recreate the data file", action='store_true')
    parser.add_argument("--paraphrase", default=False, help="Whether to generate paraphrases", action='store_true')
    parser.add_argument("--translate", default=False, help="Whether to translate paraphrases to LTL", action='store_true')
    parser.add_argument("--evaluate", default=False, help="Whether to evaluate equivalence of translated LTLs with correct LTL", action='store_true')
    parser.add_argument("--lang2ltl", default=False, help="Whether to translate utterances with Lang2LTL", action='store_true')
    args = parser.parse_args()
    with open('LTL_experiment.txt', 'w') as f:
        f.write('# LTL Experiment Log\n')
        f.write(f"Model: {args.model}\n")
        f.write(f"Create data: {args.create_data}\n")
        f.write(f"Paraphrase: {args.paraphrase}\n")
        f.write(f"Translate: {args.translate}\n")
        f.write(f"Evaluate: {args.evaluate}\n")
        f.write(f"Lang2LTL: {args.lang2ltl}\n")
        f.write(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if args.paraphrase or args.translate:
        client = OpenAI()
        model = args.model
    if args.create_data:
        create_data()
    else:
        load_data()
    if args.paraphrase:
        # Collect paraphrased utterances
        for d in data:
            utterance = d["utterance"]
            paraphrased = data_augmentation(client=client, model=model, utterance=utterance)
            d["paraphrased_utterances"] = paraphrased
    if args.translate:
        # Translate paraphrased utterances
        for d in data:
            paraphrased_utterances = d["paraphrased_utterances"]
            d["translated_ltls"] = []
            for p in paraphrased_utterances:
                translated_ltl = lang_to_ltl(client=client, model=model, utterance=p)
                d["translated_ltls"].append(translated_ltl)
    if args.evaluate:
        # Evaluate equivalence of translated LTLs with correct LTL
        for d in data:
            correct_ltl = d["formula"]
            translated_ltls = d["translated_ltls"]
            num_symbols = d["num_symbols"]
            success_count = 0
            total_count = len(translated_ltls)
            for translated_ltl in translated_ltls:
                print(f"Checking equivalence between translated LTL: {translated_ltl} and correct LTL: {correct_ltl}")
                equivalence = check_equivalence(translated_ltl, correct_ltl, num_symbols)
                if equivalence:
                    success_count += 1
            success_rate = success_count / total_count
            print(f"Paraphrase success rate for formula '{correct_ltl}': {success_rate:.2f}")
            d["paraphrased_success_rate"] = success_rate
    if args.lang2ltl:
        for d in data:
            nl_utterances = d['natural_language_utterances']
            ltls = []
            for nlu in nl_utterances:
                ltl = translate(nlu)
                ltls.append(ltl)
                print(f'NL Utt: {nlu} > LTL: {ltl}')
            d['lang2ltl_translations'] = ltls
    # Store data
    store_data()
    with open('LTL_experiment.txt', 'a') as f:
        f.write(f"Experiment ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    