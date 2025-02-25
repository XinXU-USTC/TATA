import os
import argparse
import time
import logging
import json
import re
from sympy import (
    E,
    FiniteSet,
    I,
    Intersection,
    Interval,
    Matrix,
    N,
    Union,
    pi,
    simplify,
    sqrt,
    trigsimp,
    sympify
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.exceptions import SymPyDeprecationWarning
from tqdm import tqdm
from typing import Any, Callable
import math

def get_logger(output_filename: str):
    logger = logging.getLogger(name="annotation")
    logger.setLevel(logging.INFO)

    # create directory if neccessary
    dir_name = os.path.dirname(output_filename)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    file_handler = logging.FileHandler(filename=output_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s >> %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)   # log to file and print to console
    return logger


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nowtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def save2jsonl(name, data):
    with open(name, "w") as file:
        for dict_obj in data:
            json_str = json.dumps(dict_obj)
            file.write(json_str + "\n")


def readjsonl2list(name):
    data = []  # Create an empty list to store the dictionaries

    with open(name, "r") as file:
        for line in file:
            dict_obj = json.loads(line)
            data.append(dict_obj)
    return data

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Ensure that the data is a list
            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file {file_path} does not contain valid JSON.")
    except ValueError as e:
        print(e)

def delete_extra_zero(n):
    """删除小数点后多余的0"""
    try:
        n = float(n)
    except Exception:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")  # 删除小数点后多余的0
        n = (
            int(n.rstrip(".")) if n.endswith(".") else float(n)
        )  # 只剩小数点直接转int，否则转回float
        n = str(n)
        return n

def norm_str2bool(s: str) -> bool:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    # TODO: deal with OL with boolean
    if s in ['T', 'Y']:
        return True
    elif s in ['F', 'N']:
        return False
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None
    
def latex2sympy_fix(s: str):
    sp_symbol = parse_latex(s)

    if "," in s:
        first_term = None
        try:
            first_term = parse_latex(s.split(",")[0])
        except Exception:
            pass
        if sp_symbol == first_term:
            raise LaTeXParsingError(f"{s} != {first_term}")

    return sp_symbol


def latex2sympy_interval(s: str):
    """Parse LaTeX expression like (-\\infty,0] as SymPy Interval object."""
    s = s.replace(" ", "")

    if "\\cup" in s:
        exps = s.split("\\cup")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Union(*intervals)

    if "\\cap" in s:
        exps = s.split("\\cap")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Intersection(*intervals)

    if s.startswith("\\{") and s.endswith("\\}"):
        return FiniteSet(simplify(latex2sympy_fix(s[2:-2])))
    elif s.startswith("{") and s.endswith("}"):
        return FiniteSet(simplify(latex2sympy_fix(s[1:-1])))

    if s.startswith("("):
        left_open = True
        s = s[1:]
    elif s.startswith("\\("):
        left_open = True
        s = s[2:]
    elif s.startswith("["):
        left_open = False
        s = s[1:]
    elif s.startswith("\\["):
        left_open = False
        s = s[2:]
    else:
        raise ValueError(f"Invalid interval: {s}")

    if s.endswith(")"):
        right_open = True
        s = s[:-1]
    elif s.endswith("\\)"):
        right_open = True
        s = s[:-2]
    elif s.endswith("]"):
        right_open = False
        s = s[:-1]
    elif s.endswith("\\]"):
        right_open = False
        s = s[:-2]
    else:
        raise ValueError(f"Invalid interval: {s}")

    left, right = s.split(",")
    left = simplify(latex2sympy_fix(left))
    right = simplify(latex2sympy_fix(right))
    if left.is_comparable and right.is_comparable and left >= right:
        raise ValueError(f"Invalid interval: {left}, {right}")
    interval = Interval(left, right, left_open, right_open)

    return interval


STRIP_STRS = [
    ":",
    ".",
    "/",
    ",",
    "#",
    "?",
    "$",
    '"',
    "'",
    # "ки" is the delimeter for Math-Shepherd
    "к",
    "и",
    # LaTeX
    "\\(",
    "\\)",
    "\\[",
    "\\]",
]
NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS
NO_PRECEDING_PUNCS = ["!", ")", "]", "}", "\\\\"] + STRIP_STRS

PAREN_MAP = {
    r"\(": r"\)",
    r"\[": r"\]",
    r"\{": r"\}",
    "(": ")",
    "[": "]",
    "{": "}",
}

DATETIME_FMTS = [
    # Date formats
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    # Date and time formats
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M",
    # Time formats only
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",  # 24-hour and 12-hour formats
]

BASIC_FN_NAMES = (
    "sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|coth|sech|csch|log|ln|exp|arcsin|arccos|arctan|arcsec|arccsc|arccot|arcsinh|arccosh|arctanh|arcsech|arccsch|arccoth"
).split("|")

UNITS = [
    "hour",
    "minute",
    "min",
    "sec",
    "s",
    "second",
    "day",
    "week",
    "month",
    "year",
    "meter",
    "mile",
    "kg",
    "mg",
    "g",
    "t",
    "ton",
    "nm",
    "pm",
    "um",
    "μm",
    "m",
    "cm",
    "mm",
    "dm",
    "km",
    "kilometer",
    "inch",
    "feet",
    "ft",
    "piece",
    "bit",
    "hz",
    "Hz",
    "m/s",
    "km/s",
    "m/(min^2)",
    "billion",
    "eV",
    "V",
    "C",
    "s",
    "rad",
    "rad/min",
    "in",
    "cm^3",
    "V/h",
    "m^2",
    "L/min",
    "mi/hr",
    "lb",
    r"a\.?m\.?",
    r"(?<!\\)p\.?m\.?",  # 1\pm\sqrt{5}
]

def has_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False


def is_querying4set(query):
    return "ind the" in query or ("all" in query and "separate" in query)


NDAYS_PER_WEEK = 7
WEEKDAY_ABBRS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAY_FULLS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def norm_str2weekday(s: str) -> str:
    """Converts a string representation of a weekday to its normalized form. Returns `None` if the input is not a valid weekday"""
    s = str(s).lower().strip()
    if " " in s:  # not a word
        return None

    for i_day in range(NDAYS_PER_WEEK):
        if s.startswith(WEEKDAY_ABBRS[i_day]):
            return WEEKDAY_FULLS[i_day].capitalize()
    return None




def norm_deg(s: str) -> str:
    """Normalize expressions including degrees, except independent <num>\\circ"""
    s = s.replace("rad", "")
    s = re.sub(r"^(\d+) ?\^?\\?circ$", r"\1", s)
    s = re.sub(r"(\d+) ?\^?\\?circ", r"{\1*\\frac{\\pi}{180}}", s)

    return s


def is_set(s: str):
    return (
        re.search(r"[^a-z]or(x|[^a-z])", s) is not None
        or (s.startswith("{") and s.endswith("}"))
        or (s.startswith("\\{") and s.endswith("\\}"))
    )


def fix_sqrt(
    s: str,
) -> str:
    """Fixes the formatting of square root expressions in a given string."""
    _s = re.sub(r"\\?sqrt[\(\{\[](\w+)[\)\}\]]", r"\\sqrt{\1}", s)
    _s = re.sub(r"\\?sqrt\s*(\d+)", r"\\sqrt{\1}", _s)
    return _s


def fix_fracs(s: str) -> str:
    """Fixes the formatting of fractions in a given string."""
    substrs = s.split("\\frac")
    _s = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            _s += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                _s += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return s
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        _s += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}" + b + post_substr
                    else:
                        _s += "{" + a + "}" + b
    return _s


def fix_a_slash_b(s: str) -> str:
    """
    Fixes the formatting of fractions in a given string using regular expressions.
    """
    # Define a regular expression to match fractions. Here we match two parts: the numerator (a) and the denominator (b).
    # The numerator and denominator can be numbers (\d+) or expressions containing sqrt (sqrt\(.*?\)).
    fraction_pattern = r"(\b\d+\..*|sqrt\(.*?\))\/(\d+\..*|sqrt\(.*?\)\b)"

    # Use `re.sub` to replace the matched fractions with properly formatted fractions.
    result = re.sub(
        fraction_pattern, lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", s
    )

    return result

def fix_inv_func(s: str) -> str:
    func_list = "arcsin|arccos|arctan|arcsec|arccsc|arccot|arcsinh|arccosh|arctanh|arcsech|arccsch|arccoth".split("|") 
    conv_list = "asin|acos|atan|asec|acsc|acot|asinh|acosh|atanh|asech|acsch|acoth".split("|")
    for c, f in zip(conv_list, func_list):
        s = s.replace(c, f)
    return s


STR2NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def rm_latex_env(s: str, env: str) -> str:
    """Remove LaTeX environment from a string.

    Parameters
    ----------
    s : str
        The input string.
    env : str
        The LaTeX environment name to remove.

    Returns
    -------
    str
        The string with the specified LaTeX environment removed.
    """
    s = s.replace(f"\\begin{{{env}}}", "")
    s = s.replace(f"\\end{{{env}}}", "")
    return s


LATEX_CMDS = [
    "\\textbf",
    "\\textit",
    "\\textsl",
    "\\texttt",
    "\\textsc",
    "\\textsf",
    "\\textrm",
    "\\mathrm",
    "\\mathbf",
    "\\mathit",
    "\\mathsf",
    "\\mathtt",
    "\\mathbb",
    "\\mathcal",
    "\\mathscr",
    "\\mathfrak",
    "\\bm",
    "\\em",
    "\\emph",
    "\\underline",
    "\\overline",
    "\\tiny",
    "\\scriptsize",
    "\\footnotesize",
    "\\small",
    "\\normalsize",
    "\\large",
    "\\Large",
    "\\LARGE",
    "\\huge",
    "\\Huge",
    "\\newline",
    "\\par",
    "\\noindent",
    "\\indent",
    "\\footnote",
    "\\cite",
    "\\ref",
    "\\label",
    "\\textsuperscript",
    "\\textsubscript",
    "\\text",
    "\mbox",
    "\\renewcommand{\\arraystretch}",
]

LATEX_FMT_ENVS = [
    # Align
    "align",
    "align*",
    "center",
    "flushleft",
    "flushright",
]
LATEX_LIST_ENVS = [
    "itemize",
    "enumerate",
    "description",
]


SIMPLE_RM_STRS = [
    "\n",
    "\t",
    "approximately",
    "'",
    '"',
    "\\$",
    "$",
    "￥",
    "£",
    "€",
    "{,}",
    "\\!",
    "\\,",
    "\\:",
    "\\;",
    "\\quad",
    "\\qquad",
    "\\space",
    "\\thinspace",
    "\\medspace",
    "\\thickspace",
    "~,",
    "\\ ",
    # Note the order
    "\\\\%",
    "\\%",
    "%",
    "\\left",
    "\\right",
    "^{\\circ}",
    "^\\circ",
]

SIMPLE_REPLACE_MAP = {
    "∪": "\\cup",
    "π": "\\pi",
    "∞": "\\infty",
    "∈": "\\in",
    "∩": "\\cap",
    "−": "-",
    "\\item": ",",
    "and": ",",
    ";": ",",
    "infinity": "\\infty",
    "+\\infty": "\\infty",
    "tfrac": "frac",
    "dfrac": "frac",
    "\\approx": "=",
    "\\times": "*",
    "\\cdot": "*",
    "{.": "{0.",  # "{0." equivalent to "{."
    " .": " 0.",  # " 0." equivalent to " ."
    ":": "/",  # Ratio like 3:2
}

def split_by_comma(expr: str):
    # Splits expressions by commas outside of brackets
    # 用于处理逗号的嵌套情况
    # 例子: "f(x, y, z), g(a, b, c), h(i, j)"
    in_bracket_num = 0 # 这个值为0时，说明当前不在括号内部
    splitted_expr = []
    start_idx = 0
    for i, char in enumerate(expr):
        if char in ["(", "["]:
            in_bracket_num += 1
        elif char in [")", "]"]:
            in_bracket_num -= 1
        elif char == "," and in_bracket_num == 0:
            splitted_expr.append(expr[start_idx:i].strip())
            start_idx = i + 1

    if start_idx < len(expr):
        splitted_expr.append(expr[start_idx:].strip())  
            
    if splitted_expr:
        splitted_expr = [item.strip("$").strip() for item in splitted_expr] 
        
    return splitted_expr

def norm_ans_str(ans: str) -> str:
    """Normalize answer string for **all kinds** of answers."""
    #ans_list = split_by_comma(ans.strip("()"))
    if len(ans) == 0:
        return ans
    if ans[0] == '(' and ans[-1] == ')':
        ans_list = split_by_comma(ans.strip("()"))
    elif ans[0] == '{' and ans[-1] == '}':
        ans_list = split_by_comma(ans.strip("{}"))
    else:
        ans_list = split_by_comma(ans)
    new_ans_list = []

    for ans_i in ans_list:
        ans_item = str(ans_i)
        ans_item = ans_item.replace("\n", "")
        ans_item = ans_item.strip()

        # remove impropriate trailing punctuations
        ans_item = clean(ans_item)

        # bool
        ans_bool = norm_str2bool(ans_item)
        if ans_bool is not None:
            new_ans_list.append(str(ans_bool))
            continue

        # weekdays
        ans_weekday = norm_str2weekday(ans_item)
        if ans_weekday is not None:
            new_ans_list.append(str(ans_weekday))
            continue

        # math normalize
        ans_item = norm_math_str(ans_item)
        new_ans_list.append(ans_item)

    assert len(ans_list) == len(new_ans_list)

    if len(new_ans_list) == 1:
        return new_ans_list[0]
    return "(" + ", ".join(new_ans_list) + ")"

def norm_pm(s: str) -> str:
    """Replaces the LaTeX symbols '$1\\pm$2' or '$1\\mp$2' with '$1-$2,$1+$2'."""

    def replace_pm(match):
        # Extracts the first and second parts of the match.
        first_part, second_part = match.groups()
        # Creates the replacement string as specified.
        return f"{first_part}-{second_part},{first_part}+{second_part}"

    _s = remove_out_paren(s)
    pattern = r"([\w\.\\{}\+\-\*\^]+?)(?:\\pm|\\mp)([\w\.\\{}\+\-\*\^]+)"

    if re.search(pattern, _s):
        # Use re.sub to replace all occurrences of the pattern in the input string.
        return re.sub(pattern, replace_pm, _s)
    else:
        return s
    


def eq(self, ref: str, ans: str) -> bool:
    """Check if reference answer and prediction answer are **literally** equal."""
    return ref == ans


