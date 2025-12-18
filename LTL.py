from openai import OpenAI
import time
from NeuralRewardMachines.LTL_tasks import formulas, utterances

client = OpenAI()

models = ['gpt-5.2', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano']

models = models[:]

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