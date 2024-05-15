# -*- coding: utf-8:
from collections import namedtuple
import numpy as np
import sympy

__doc__="""
Expr_with_args class and helper functions.

Expr_with_args class represents a function of form f(x; param1, param2, ..., paramN)
i.e. a function with one independent variable and parameters.
"""

#### define named tuple for represent (expression, arg symbols, arg names)
class Expr_with_args(namedtuple('Expr_with_args', 'expr, arg_symbols, arg_names')):
    """
    A class to represent a function of form f(x, param1, param2, ..., paramN)
    """
    def __call__(self, x, *args, **kws):
        return EWA_to_func(self)(x, *args, **kws)

#### an example of EWA definition
def make_EWA_linear():
    """
    An example routine to make a EWA.

    :return: a EWA of a linear function :math:`f(x, a, b) = ax + b`

    - **x** is independent variable, and the rest (**a, b**) are parameters.

    (This routine is also intended as a template to create other functions.)
    """

    ## define argument symbols and its name(usable as python variable name).
    arg_latex = r'a b'
    arg_names = r'a b'.split()
    #arg_names = arg_latex.split()

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    a, b        = arg_symbols

    ## define other symbols to use
    x   = sympy.symbols('x') # for dependent variable
    I   = sympy.I            # imagnary unit
    pi  = sympy.pi           # pi
    exp = sympy.exp          # exponential

    ## define expression using symbols to use
    expr = a*x + b

    return Expr_with_args(expr, arg_symbols, arg_names)

linear_ewa = make_EWA_linear()
# => Expr_with_args(expr=a*x + b, arg_symbols=(a, b), arg_names=['a', 'b'])

#### functions to convert EWA to function

def EWA_to_func(EWA):
    """
    Convert EWA with argument (x, param1, param2, ..., paramN) to a function.

    :param EWA: a Expr_with_args of a function, whose arguments are 1 depdendent variable and other paremeters.

    :return: a function of form f(x, param1, param2, ..., paramN) which calculates the value of expression at given x and parameters.

    ::

        Example:
        >>> linear_func = EWA_to_func(linear_ewa)
        >>> linear_func(10, 3, 5)
        35
        >>> linear_func(np.array([10, 20, 30]), 3, 5)
        array([35, 65, 95])
    """
    expr, arg_symbols, arg_names = EWA
    expr_ = expr.subs(zip(arg_symbols, arg_names))
    args = sympy.symbols(['x']+arg_names)
    func = sympy.lambdify(args, expr_, 'numpy', dummify=False)
    return func


def EWA_to_gradfunc(EWA, useargs=None):
    u"""
    Convert EWA with argument (x, param1, param2, ..., paramN) to function

    :param EWA: a Expr_with_args of a function, whose arguments are 1 depdendent variable and other paremeters.
    :param useargs: list of parameter name, if you need only subset of gradient. if None (default), return gradient about all parameters.

    :return: a function whose argument is (x, param1, param2, ..., paramN) and returns a array of form np.array([∂f(x, param1, param2, ..., paramN) / ∂(param1), ∂f(x, param1, param2, ..., paramN) / ∂(param2),..., ∂f(x, param1, param2, ..., paramN) / ∂(paramN)]).

    ::

        Example:
        >>> linear_gradfunc = EWA_to_gradfunc(linear_ewa)
        >>> linear_gradfunc(10, 3, 5)
        array([10,  1])
        >>> linear_gradfunc(np.array([10, 20, 30]), 3, 5)
        array([[10, 20, 30], [ 1,  1,  1]])
    """
    expr, arg_symbols, arg_names = EWA

    if useargs:
        _arg_symbols = []
        _arg_names   = []
        for i, n in enumerate(arg_names):
            if n in useargs:
                _arg_symbols.append(arg_symbols[i])
                _arg_names.append(arg_names[i])
    else:
        _arg_symbols = arg_symbols
        _arg_names   = arg_names

    funcargs = ['x'] + arg_names
    gradfuncs = [sympy.lambdify(funcargs,
                                sympy.diff(expr, var).subs(zip(arg_symbols, arg_names)),
                                'numpy', dummify=False)
                 for var in _arg_symbols]


    lambdastr = ("lambda %s: np.array(np.broadcast_arrays(%s))" %
                    (', '.join(funcargs),
                     ', '.join(['gradfuncs[%d](%s)' % (i, ', '.join(funcargs))
                                for i in range(len(_arg_names))])))
    #print( lambdastr )
    ns = globals()
    ns.update(locals())
    func = eval(lambdastr, ns)

    return func

from itertools import chain
def get_symbols(expr):
    """
    :return: a list of variables in given sympy expression
    """
    if isinstance(expr, sympy.Symbol):
        return [expr]
    else:
        return list(chain.from_iterable([get_symbols(arg) for arg in expr.args]))

from functools import wraps
def memoize(f):
    cache = dict()
    @wraps(f)
    def wrapper(arg):
        if arg not in cache:
            cache[arg] = f(arg)
        return cache[arg]
    return wrapper

@memoize
def EWA_from_string(string_expr):
    expr = sympy.sympify(string_expr)
    symbols = get_symbols(expr)
    independent_arg = 'x'
    independent_arg_sym = sympy.Symbol(independent_arg)
    print( symbols, independent_arg_sym, map(type, symbols) )
    if independent_arg_sym in symbols:
       args = [s for s in symbols if s != independent_arg_sym]
       args.sort(key=lambda x: x.name)
    else:
        raise RuntimeError("'x' not found in given expr")
    return Expr_with_args(expr, args, map(lambda x:x.name, args))
