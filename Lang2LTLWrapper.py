import logging
from Lang2LTL.lang2ltl import lang2ltl
from flloat.parser.ltlf import LTLfParser

# Set up logging to print INFO messages to the console
logging.basicConfig(level=logging.INFO)

utt = "Visit the pickaxe and go to the lava."
obj2sem = {
    "pickaxe": {},
    "lava": {},
    "door": {},
    "gem": {}
}
keep_keys = ["pickaxe", "lava", "door", "gem"]

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

def lang2ltl_translate(utt):
    result = lang2ltl(utt, obj2sem, keep_keys)
    # print("LTL formula:", result)
    infix = prefixToInfix(result)
    # print(f'Infix: {infix}')
    # parser = LTLfParser()
    # formula = parser(infix)
    formula = infix
    return formula