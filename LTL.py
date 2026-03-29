import argparse
from openai import OpenAI
import time
from NeuralRewardMachines.LTL_tasks import formulas, utterances
from DFA.DFA import DFA
import json
from datetime import datetime
from Lang2LTLWrapper import toSymbolic, translate
import spot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import unique_ordered_list
import numpy as np

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
'Use & as AND operator, | as OR, -> as implication, ! as NOT. Use F as finally temporal operator, ' \
'G as globally temporal operator. Use c0 as symbol for pickaxe, c1 as symbol for lava, c2 as symbol ' \
'for door, c3 as symbol for gem. Only provide the LTL task as your response,  without the triple ' \
'backtick tag defining a code block. ' \

prompt = "You are given a natural language instruction for a task in a simulated environment. " \
"Your goal is to convert the instruction into a Linear Temporal Logic (LTL) formula. " \
"Use & as AND operator, | as OR, -> as implication, ! as NOT. Use F as finally temporal operator, " \
"G as globally temporal operator. Use c0 as symbol for pickaxe, c1 as symbol for lava, c2 as symbol " \
"for door, c3 as symbol for gem. Remember to not assume and impose temporal constraints if those are not " \
"explicitly expressed in the instruction. Only provide the LTL formula as your response, without the triple " \
"backtick tag defining a code block. "

prompt_COT = 'You are given a natural language instruction for a task in a simulated environment. ' \
'Your goal is to convert the instruction into a Linear Temporal Logic (LTL) formula. ' \
'Use & as AND operator, | as OR, -> as implication, ! as NOT. Use F as finally temporal operator, ' \
'G as globally temporal operator. Use c0 as symbol for pickaxe, c1 as symbol for lava, c2 as symbol ' \
'for door, c3 as symbol for gem. Remember to not assume and impose temporal constraints if those are not ' \
'explicitly expressed in the instruction. Think and perform your considerations step by step. ' \
'As final sentence of your response, provide the translated LTL task, enclosed in "**". ' \

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
    input = prompt + "\n\nInstruction: " + utterance
    response = client.responses.create(
        model=model,
        input=input
    )
    paraphrased = response.output_text.split('\n')
    return paraphrased

def lang_to_ltl(client, model='gpt-5.2', utterance="Visit the pickaxe and visit the lava.", cot=False):
    task = prompt_COT if cot else prompt
    input = task + "\n\nInstruction: " + utterance
    response = client.responses.create(
        model=model,
        input=input
    )
    if cot:
        print("Model's step-by-step considerations:\n", response.output_text)
        ltl = response.output_text.split('**')[-2].strip()
    else:
        ltl = response.output_text
    return ltl

def check_equivalence(ltl1, ltl2, num_symbols):
    try:
        f1 = spot.formula(ltl1)
        f2 = spot.formula(ltl2)
    except Exception as e:
        print(f"Error parsing LTL formulas: {e}")
        return False
    equivalence = spot.are_equivalent(f1, f2)
    # equivalence_formula = f"(({ltl1}) -> ({ltl2})) & (({ltl2}) -> ({ltl1}))"
    # equivalence_formula = f"({ltl1}) <-> ({ltl2})"
    # equivalence_automaton = DFA(ltl_formula=equivalence_formula,
    #                                     num_symbols=num_symbols, 
    #                                     formula_name="equivalence_check", 
    #                                     dictionary_symbols=dictionary_symbols)
    # # equivalence_automaton.write_dot_file(f"data/equivalence_check_{i}.dot")
    # equivalence = (equivalence_automaton.num_of_states == 1 and equivalence_automaton.acceptance[0] == True)
    return equivalence

def plot_equivalence_results(data):
    df = pd.DataFrame(data)
    formulas = df['formula'].tolist()
    accuracies = {}
    accuracies['pr_to_lang2ltl'] = df['pr_lang2ltl_accuracy_spot'].tolist()
    accuracies['nlu_to_lang2ltl'] = df['natural_language_utterances_to_Lang2LTL_accuracy'].tolist()
    #accuracies['gpt-5.2-CoT'] = df['poll_translations_to_GPT5-COT_accuracy'].tolist()
    colors = sns.color_palette("Greys", n_colors=len(accuracies))
    plt.figure(figsize=(12, 6))
    x = np.arange(len(formulas))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in accuracies.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Accuracy')
    ax.set_title('Translation Success Rate by LTL Formula')
    ax.set_xticks(x + width, formulas)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/Lang2LTL/PR_NLU_Translation_Success_Rate.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle LTL tasks with LLMs")
    parser.add_argument("--model", type=str, default='gpt-5.2', help="Language model to use")
    parser.add_argument("--create-data", default=False, help="Whether to recreate the data file", action='store_true')
    parser.add_argument("--paraphrase", default=False, help="Whether to generate paraphrases", action='store_true')
    parser.add_argument("--translate", default=False, help="Whether to translate paraphrases to LTL", action='store_true')
    parser.add_argument("--cot", default=False, help="Whether to use chain-of-thought prompting", action='store_true')
    parser.add_argument("--lang2ltl", default=False, help="Whether to translate utterances with Lang2LTL", action='store_true')
    parser.add_argument("--to-symbolic", default=False, help="Convert translated LTL formulas to symbolic form", action='store_true')
    parser.add_argument("--filter", default=False, help="Filter translated LTL formulas", action='store_true')
    parser.add_argument("--evaluate", default=False, help="Whether to evaluate equivalence of translated LTLs with correct LTL", action='store_true')
    parser.add_argument("--plot", default=False, help="Plot translation success rates", action='store_true')
    args = parser.parse_args()
    with open('LTL_experiment.txt', 'w') as f:
        f.write('# LTL Experiment Log\n')
        f.write(f"Model: {args.model}\n")
        f.write(f"Create data: {args.create_data}\n")
        f.write(f"Paraphrase: {args.paraphrase}\n")
        f.write(f"Translate: {args.translate}\n")
        f.write(f"COT: {args.cot}\n")
        f.write(f"Evaluate: {args.evaluate}\n")
        f.write(f"Lang2LTL: {args.lang2ltl}\n")
        f.write(f"Plot: {args.plot}\n")
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
        cot = args.cot
        input_label = "poll_translations"
        output_label = input_label + "_to_GPT5" + ("-COT" if cot else "")
        # Translate paraphrased utterances
        for d in data:
            utterances = d[input_label]
            for p in utterances[18:]:
                ltl = lang_to_ltl(client=client, model=model, utterance=p, cot=cot)
                print(f'Paraphrased Utt: {p} > LTL: {ltl}')
                d[output_label].append(ltl)
    if args.lang2ltl:
        input_label = "poll_translations"
        output_label = input_label + "_to_Lang2LTL"
        for d in data:
            nl_utterances = d[input_label]
            for nlu in nl_utterances[18:]:
                ltl = translate(nlu)
                d[output_label].append(ltl)
                print(f'NL Utt: {nlu} > LTL: {ltl}')
    if args.filter:
        for d in data:
            ground_truth = d['formula']
            translated_ltls = unique_ordered_list(d["symbolic_lang2ltl_translations"])
            filtered_ltls = []
            for ltl in translated_ltls:
                if not check_equivalence(ltl, ground_truth, d['num_symbols']):
                    filtered_ltls.append(ltl)
            d['filtered_symbolic_lang2ltl_translations'] = filtered_ltls
    if args.evaluate:
        # Evaluate equivalence of translated LTLs with correct LTL
        input_label = "natural_language_utterances_to_GPT5-COT"
        output_label = input_label + "_accuracy"
        for d in data:
            correct_ltl = d["formula"]
            translated_ltls = d[input_label]
            num_symbols = d["num_symbols"]
            success_count = 0
            total_count = len(translated_ltls)
            for translated_ltl in translated_ltls:
                print(f"Checking equivalence between translated LTL: {translated_ltl} and correct LTL: {correct_ltl}")
                equivalence = check_equivalence(translated_ltl, correct_ltl, num_symbols)
                if equivalence:
                    success_count += 1  
            success_rate = success_count / total_count if total_count > 0 else 0
            print(f"Paraphrase success rate for formula '{correct_ltl}': {success_rate:.2f}")
            d[output_label] = success_rate
    if args.plot:
        plot_equivalence_results(data)
    # Store data
    store_data()
    with open('LTL_experiment.txt', 'a') as f:
        f.write(f"Experiment ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    