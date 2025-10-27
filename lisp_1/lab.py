"""
6.101 Lab:
LISP Interpreter Part 1
"""

#!/usr/bin/env python3

import sys

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass

class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def tokenize_comments(source):
    """Handles comments and line breaks and returns a new string of expression with comments removed."""
    # Split  source into lines and remove comments
    source_w_line_breaks = source.split('\n')
    lines_wo_comments = []

    for line in source_w_line_breaks:
        # Split line at first occurrence of ';' -- take first part
        #also strip whitespace from ends
        cleaned_line = line.split(';', 1)[0].strip()
        if cleaned_line:
            lines_wo_comments.append(cleaned_line)

    # Join cleaned lines with space
    return ' '.join(lines_wo_comments)

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    source2 = tokenize_comments(source)

    # Initialize an empty list to store tokens
    tokens = []
    token = ""

    # Iterate through each character in the cleaned source
    for char in source2:
        if char in '()':
            # If there's an accumulated token, append it to the list
            #and reset the token to empty str for next one
            if token:
                tokens.append(token)
                token = ""
            # then Append the special character -- either '(' or ')' --  to the list
            tokens.append(char)
        elif char.isspace():
            # If whitespace character is found, append aggregated token to the list
            if token:
                tokens.append(token)
                token = ""
        else:
            # Accumulate the current character into the token
            token += char

    # Append any remaining token after the loop
    if token:
        tokens.append(token)

    return tokens

def parse(tokens):
    def parse_exp(index):
        """Parse an expression starting at a given index in the tokens list."""

        # If current token is an opening parenthesis
        if tokens[index] == '(':
            parsed = []
            index += 1
            while index < len(tokens) and tokens[index] != ')':
                # Recursively parse the next expression
                item, index = parse_exp(index)
                parsed.append(item)
            return parsed, index + 1

        else:
            # Return a single token as number or symbol
            return number_or_symbol(tokens[index]), index + 1

    # Parse the input tokens starting from index 0
    parsed_expression, next_index = parse_exp(0)

    return parsed_expression


######################
# Built-in Functions #
######################

def multiply(args):
    prod = 1
    for factor in args:
        prod *= factor
    return prod
def divide(args):
    if not args:
         raise ValueError('args is empty list')
    elif len(args) == 1:
        return 1/args[0]
    else:
        quotient = args[0]
        for divisor in args[1:]:
            quotient /= divisor
        return quotient

class Frame():
    def __init__(self,parent = None):
        self.mappings = {}
        self.parent = parent
    def __setitem__(self,var,val):
        self.mappings[var] = val
    def __contains__(self, vari):
        """checks if var is defined in current frame or not"""
        if vari in self.mappings:
            return True
        elif self.parent is None:
            return False
        else:
            return vari in self.parent
    def delete(self,var):
        """removes a variable from mappings in frame"""
        if var not in self.mappings:
            raise SchemeNameError('var not defined in this frame')
    def get_frame(self,vari):
        if vari not in self:
            raise SchemeNameError('var nonexistent in any frame')
        if vari in self.mappings:
            return self
        else:
            return self.parent.get_frame(vari)

    def __getitem__(self,var):
        if var in self.mappings:
            return self.mappings[var]
        elif self.parent is None:
            raise SchemeNameError('var not found in any frame')
        else:
            return self.parent.__getitem__(var)


class Funcs():
    def __init__(self,args,body,frame):
        self.args = args
        self.body = body
        self.enc_frame = frame
    def __call__(self,values):
        new_frame = Frame(self.enc_frame)
        if len(values) != len(self.args):
            raise SchemeEvaluationError("inc num of args")
        for param,val in zip(self.args,values):
            new_frame.__setitem__(param,val)
            #new_frame[param] = val
        return evaluate(self.body,new_frame)
    def __str__(self):
        return f'arg: {str(self.args)}, body: {str(self.body)}'





##############
# Evaluation #
##############

def evaluate(tree, frame = None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if frame is None:
        frame = make_initial_frame()
    if isinstance(tree,(str)):
        return frame.__getitem__(tree)

    elif isinstance(tree,(float,int)):
        return tree
    elif isinstance(tree,list):
        if len(tree) == 0:
            raise SchemeEvaluationError

        first = tree[0]
        if first == "define":
            second = tree[1]
            if isinstance(second,list):
                exp = ['define',second[0],['lambda',second[1:],tree[2]]]
                return evaluate(exp,frame)
            var = second
            val = evaluate(tree[2],frame)
            frame[var] = val
            return val
        elif first == "lambda":
            params = tree[1]
            expresh = tree[2]
            return Funcs(params,expresh,frame)
            #analaogous to Functions() class
        ###Built in func handling
        if not isinstance(first,(str,list)):
            raise SchemeEvaluationError

        else:
            func = evaluate(first,frame)
            if not callable(func):
                raise SchemeEvaluationError
            evald = func([evaluate(itm,frame) for itm in tree[1:]])
            return evald

def make_initial_frame():
    base_frame = Frame()
    base_frame.mappings = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": divide,
}
    global_frame = Frame(base_frame)
    return global_frame
if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    import os
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl
    #schemerepl.SchemeREPL(use_frames=True, verbose=False).cmdloop()
    #print(evaluate(['+', 3, ['-', 7, 5]]))
    #print(evaluate(['define', 'somevariable', ['+', 1, 2]]))
    #print(evaluate(['define','spam','x']))
    #print(tokenize(""))
    print(evaluate(3))
