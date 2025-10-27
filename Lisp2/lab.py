"""
6.101 Lab:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)


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


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
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


# KEEP THE ABOVE LINES INTACT, BUT REPLACE THIS COMMENT WITH YOUR lab.py FROM


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
    source_w_line_breaks = source.split("\n")
    lines_wo_comments = []

    for line in source_w_line_breaks:
        # Split line at first occurrence of ';' -- take first part
        # also strip whitespace from ends
        cleaned_line = line.split(";", 1)[0].strip()
        if cleaned_line:
            lines_wo_comments.append(cleaned_line)

    # Join cleaned lines with space
    return " ".join(lines_wo_comments)


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
        if char in "()":
            # If there's an accumulated token, append it to the list
            # and reset the token to empty str for next one
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
        if tokens[index] == "(":
            parsed = []
            index += 1
            while index < len(tokens) and tokens[index] != ")":
                # Recursively parse the next expression
                item, index = parse_exp(index)
                parsed.append(item)
            if index >= len(tokens):
                raise SchemeSyntaxError("Open Paren Unmatched")
            return parsed, index + 1
        elif tokens[index] == ")":
            raise SchemeSyntaxError("Closing Paren Unmatched")

        else:
            # Return single token as num or symbol
            return number_or_symbol(tokens[index]), index + 1

    # Parse inp tokens starting from index 0
    parsed_expression, next_index = parse_exp(0)
    if next_index < len(tokens):
        raise SchemeSyntaxError("tokens left post parsing")
    return parsed_expression


class Pair:
    def __init__(self, elem, next_ptr):
        self.car = elem
        self.cdr = next_ptr

    def set_ptr(self, new_ptr):
        self.cdr = new_ptr

    def __str__(self):
        return f"Pair({self.car},{self.cdr})"


class EmptyList:
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, EmptyList)


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
        raise ValueError("args is empty list")
    elif len(args) == 1:
        return 1 / args[0]
    else:
        quotient = args[0]
        for divisor in args[1:]:
            quotient /= divisor
        return quotient


def equal(args):
    first_itm = args[0]
    for itm in args[1:]:
        if not itm == first_itm:
            return "#f"
    return "#t"


def greater(args):
    for i in range(len(args) - 1):
        if not args[i] > args[i + 1]:
            return "#f"
    return "#t"


def less(args):
    for i in range(len(args) - 1):
        if not args[i] < args[i + 1]:
            return "#f"
    return "#t"


def gr_eq(args):
    for i in range(len(args) - 1):
        if not args[i] >= args[i + 1]:
            return "#f"
    return "#t"


def les_eq(args):
    for i in range(len(args) - 1):
        if not args[i] <= args[i + 1]:
            return "#f"
    return "#t"


def not_it(argument):
    if isinstance(argument, list):
        if len(argument) != 1:
            raise SchemeEvaluationError
        arg = argument[0]
    if arg == "#t":
        return "#f"
    else:
        return "#t"


def cons(args):
    if len(args) != 2:
        raise SchemeEvaluationError("Wrong # args to cons")

    return Pair(args[0], args[1])


def car(pair_type):
    if not isinstance(pair_type, list):
        pair_type = [pair_type]
    if len(pair_type) != 1:
        raise SchemeEvaluationError

    if not isinstance(pair_type[0], Pair):
        raise SchemeEvaluationError

    return pair_type[0].car


def cdr(pair_type):
    if not isinstance(pair_type, list):
        pair_type = [pair_type]
    if len(pair_type) != 1:
        raise SchemeEvaluationError

    if not isinstance(pair_type[0], Pair):
        raise SchemeEvaluationError

    return pair_type[0].cdr


def list2(args):
    if len(args) == 0:
        return EmptyList()

    elif len(args) == 1:
        return Pair(args[0], EmptyList())
    else:
        list_obj = Pair(args[0], list2(args[1:]))

    return list_obj


def list_q(object):
    if isinstance(object, list):
        if len(object) != 1:
            raise SchemeEvaluationError
        object = object[0]

    while isinstance(object, Pair):
        # keep grabbing first itm until you reach empty ()
        object = cdr(object)

    if object == EmptyList():
        return "#t"

    else:
        return "#f"


def len_list(pair_type, sofar=0):
    """
    Given pair object, return the length. Raise eval
    error if not given a pair object.
    """
    if isinstance(pair_type, list):
        if len(pair_type) != 1:
            raise SchemeEvaluationError
        pair_type = pair_type[0]
    if pair_type == EmptyList():
        return sofar
    if list_q(pair_type) == "#f":
        raise SchemeEvaluationError

    while pair_type != EmptyList():
        left = car(pair_type)
        right = cdr(pair_type)
        if left != EmptyList():
            sofar += 1
        pair_type = right
    return sofar


def val_at_index(li):
    if len(li) != 2:
        raise SchemeEvaluationError
    pair, idx = li
    if not isinstance(pair, Pair) or not isinstance(idx, int):
        raise SchemeEvaluationError

    def inner_val(linlis, idx):
        print(linlis)
        if not isinstance(linlis, Pair):
            raise SchemeEvaluationError
        if idx == 0:
            return linlis.car
        return inner_val(linlis.cdr, idx - 1)

    return inner_val(pair, idx)


def lis_append(args):
    if len(args) == 0:
        return EmptyList()

    elif list_q(args[0]) == "#f":
        raise SchemeEvaluationError("cant append, obj not list")

    elif args[0] == EmptyList():
        return lis_append(args[1:])

    else:
        rest = [cdr(args[0])] + args[1:]
        return Pair(car(args[0]), lis_append(rest))


class Frame:
    def __init__(self, parent=None):
        self.mappings = {}
        self.parent = parent

    def __setitem__(self, var, val):
        self.mappings[var] = val

    def __contains__(self, vari):
        """checks whether var def'd in current frame"""
        if vari in self.mappings:
            return True
        elif self.parent is None:
            return False
        else:
            return vari in self.parent

    def delete(self, var):
        """removes var from mappings in frame"""
        if var not in self.mappings:
            raise SchemeNameError("var not defd in frame")

    def get_frame(self, vari):
        if vari not in self:
            raise SchemeNameError("var nonexistent in any frame")
        if vari in self.mappings:
            return self
        else:
            return self.parent.get_frame(vari)

    def __getitem__(self, var):
        if var in self.mappings:
            return self.mappings[var]
        elif self.parent is None:
            raise SchemeNameError("var not found in any frame")
        else:
            return self.parent.__getitem__(var)


class Funcs:
    def __init__(self, args, body, frame):
        self.args = args
        self.body = body
        self.enc_frame = frame

    def __call__(self, values):
        new_frame = Frame(self.enc_frame)
        if len(values) != len(self.args):
            raise SchemeEvaluationError("inc num of args")
        for param, val in zip(self.args, values):
            new_frame.__setitem__(param, val)

        return evaluate(self.body, new_frame)

    def __str__(self):
        return f"arg: {str(self.args)}, body: {str(self.body)}"


##############
# Evaluation #
##############


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if frame is None:
        frame = make_initial_frame()
    literals = {
        "#t": "#t",
        "#f": "#f",
    }
    if isinstance(tree, (float, int, EmptyList)):
        return tree

    elif isinstance(tree, (str)):
        if tree in literals:
            return literals[tree]
        return frame.__getitem__(tree)
    
    elif isinstance(tree, list):
        if len(tree) == 0:
            return EmptyList()

        first = tree[0]

        if first == "begin":
            for expression in tree[1:]:
                output = evaluate(expression, frame)
            return output

        if first == "del":
            if len(tree) != 2:
                raise SchemeEvaluationError
            var = tree[1]
            if var not in frame.mappings:
                raise SchemeNameError
            return frame.mappings.pop(var)

        if first == "set!":
            if len(tree) != 3:
                raise SchemeEvaluationError
            var = tree[1]
            expres = tree[2]
            if not isinstance(var, str):
                raise SchemeEvaluationError
            result = evaluate(expres, frame)
            nearest_frame = frame.get_frame(var)
            nearest_frame.mappings |= {var: result}
            return result

        if first == "let":
            func_frame = Frame(frame)
            for items in tree[1]:
                if len(items) != 2:
                    raise SchemeEvaluationError
                var, val = items
                func_frame.mappings |= {var: evaluate(val, frame)}
            return evaluate(tree[2], func_frame)

        if first == "define":
            second = tree[1]
            if isinstance(second, list):
                exp = ["define", second[0], ["lambda", second[1:], tree[2]]]
                return evaluate(exp, frame)
            var = second
            val = evaluate(tree[2], frame)
            frame[var] = val
            return val

        elif first == "lambda":
            params = tree[1]
            expresh = tree[2]
            return Funcs(params, expresh, frame)
            # analaogous to Funcs() class

        elif first == "if":
            pred = tree[1]
            true_exp = tree[2]
            false_exp = tree[3]
            if evaluate(pred, frame) == "#t":
                return evaluate(true_exp, frame)
            elif evaluate(pred, frame) == "#f":
                return evaluate(false_exp, frame)

        elif first == "del":
            return frame.delete(tree[1])

        elif first == "and":
            args = tree[1:]
            for exp in args:
                if evaluate(exp, frame) == "#f":
                    return "#f"
            return "#t"

        elif first == "or":
            args = tree[1:]
            for exp in args:
                if evaluate(exp, frame) == "#t":
                    return "#t"
            return "#f"

        ###Built-in func handling
        if not isinstance(first, (str, list)):
            raise SchemeEvaluationError

        else:
            func = evaluate(first, frame)
            if not callable(func):
                raise SchemeEvaluationError
            evald = func([evaluate(itm, frame) for itm in tree[1:]])
            return evald


def make_initial_frame():
    base_frame = Frame()
    base_frame.mappings = {
        "+": sum,
        "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
        "*": multiply,
        "/": divide,
        "equal?": equal,
        ">": greater,
        ">=": gr_eq,
        "<": less,
        "<=": les_eq,
        "not": not_it,
        "list": list2,
        "list?": list_q,
        "length": len_list,
        "list-ref": val_at_index,
        "append": lis_append,
        "cons": cons,
        "car": car,
        "cdr": cdr,
    }
    global_frame = Frame(base_frame)
    return global_frame


# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.
def evaluate_file(file_name, frame=make_initial_frame()):
    with open(file_name) as f:
        data = f.read()
    return evaluate(parse(tokenize(data)), frame)


if __name__ == "__main__":
    import os

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    repl_frame = make_initial_frame()
    file_args = sys.argv[1:]
    if len(file_args) > 0:
        for file_name in file_args:
            evaluate_file(file_name, repl_frame)
    import schemerepl

    schemerepl.SchemeREPL(
        sys.modules[__name__], use_frames=True, verbose=False, repl_frame=repl_frame
    ).cmdloop()
