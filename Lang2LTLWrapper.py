import logging
from Lang2LTL.lang2ltl import lang2ltl
# from temporal_reasoning import parse
from flloat.parser.ltlf import LTLfParser
import sys

# Set up logging to print INFO messages to the console
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

utt = "Visit the pickaxe and go to the lava."
# lmk2sem = obj2sem
obj2sem = {
    "pickaxe": {"material": ["iron", "wood"], 
            "type": "tool", 
            "usage": "mining", 
            "color": ["brown", "gray"],
            "action": ["pick", "hit", "grab", "use"]
        },
    "lava": {"material": ["lava", "stone"], 
            "type": "hazard", 
            "color": ["red", "orange"], 
            "state": "liquid",
            "action": ["avoid", "step on" ]
        },
    "door": {"material": ["wood", "iron"], 
            "type": "structure", 
            "state": "closed", 
            "action": ["open", "close", "enter", "exit"]
        },
    "gem": {"material": ["diamond", "ruby"], 
            "type": "treasure", 
            "color": ["blue", "red", "green", "yellow"], 
            "value": "high", 
            "action": ["collect", "pick up"]
        }
}
keep_keys = ["pickaxe", "lava", "door", "gem"]

obj_to_symbol = {
    "pickaxe": "c0",
    "lava": "c1",
    "door": "c2",
    "gem": "c3"
}

def mapOperator(op):
    operator_map = {
        'i': '->',
    }
    map = operator_map.get(op, op)
    return map

def isBinaryOperator(c):
    # TODO: Check Lang2LTL formula syntax
    binaryOperators = ['&', '|', 'i', 'U']
    isBinaryOperator = c in binaryOperators
    return isBinaryOperator

def isUnaryOperator(c):
    # TODO: Check Lang2LTL formula syntax
    unaryOperators = [ '!', 'X', 'F', 'G']
    isUnaryOperator = c in unaryOperators
    return isUnaryOperator

def prefixToInfix(prefix):
    ''' Convert formula in prefix form to infix form.'''
    stack = []
    # read prefix in reverse order
    ops = prefix.split(' ')
    ops.reverse()
    for op in ops:
        if isUnaryOperator(op):
            formula = f'({mapOperator(op)} {stack.pop()})'
            stack.append(formula)
        elif isBinaryOperator(op):
            formula = f'({stack.pop()} {mapOperator(op)} {stack.pop()})'
            stack.append(formula)
        else:
            stack.append(op)
        # print(stack)
    infix = stack.pop()
    return infix

def toSymbolic(formula):
    ''' Convert Lang2LTL formula to symbolic form used in LTLfParser.'''
    for sym, obj in obj_to_symbol.items():
        formula = formula.replace(sym, obj)
    return formula

def translate(utt):
    result = lang2ltl(utt, obj2sem, keep_keys, rer_prompt_fpath='data/Lang2LTL/RErecognition/rer_prompt_GWE.txt')
    # result = parse(utt, obj2sem, keep_keys, rer_prompt_fpath='data/rer_prompt_GWE.txt')
    # print("LTL formula:", result)
    infix = prefixToInfix(result)
    # print(f'Infix: {infix}')
    # parser = LTLfParser()
    # formula = parser(infix)
    symbolic_formula = toSymbolic(infix)
    return symbolic_formula