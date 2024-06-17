import collections
import inspect
from copy import deepcopy
import inspect

import scipy.stats
import numpy as np

#import ad
import lmfit

from .expr_with_args import Expr_with_args, EWA_to_func, EWA_to_gradfunc, EWA_from_string
from .fitresult import FitResult

try:
    import sympy
except ImportError:
    sympy_installed = False
else:
    sympy_installed = True

def fit(function, xs, ys, err=None, via=None, range=None, silent=False, convert=None, **kws):
    return _fit(function, xs, ys, err, via, range, silent, convert, **kws)

def _fit(function, xs, ys, err=None, via=None, range=None, silent=False,
         convert=None, **kws):
    """
    hoge
    """

    ## get function
    if type(function) == Expr_with_args:
        func = EWA_to_func(function)
    elif callable(function):
        func = function
    elif isinstance(function, (str, unicode)):
        function = EWA_from_string(function)
        func = EWA_to_func(function)
    else:
        raise ValueError("can't get function of %s" % (function,))

    # get its argument
    if type(function) == Expr_with_args:
        tmpargs = function.arg_names
    else:
        argspec  = inspect.getfullargspec(func)
        tmpargs = argspec.args[1:]

    ## prepare params argument
    params = lmfit.Parameters()
    for k in tmpargs:
        params.add(k, value=1)

    if via is None:
        pass
    elif isinstance(via, lmfit.Parameters):
        params = via
    elif isinstance(via, dict):
        for k, v in via.items():
            if isinstance(v, (int, float, complex)):
                params[k].value=v
            elif isinstance(v, lmfit.Parameter):
                params[k] = v
            else:
                raise RuntimeError('%s: Not Implemented parameter value type' % type(v))
    else:
        raise RuntimeError('%s: Not Implemented via type' % type(via))

    ## prepare Dfun if function is an Expr_with_args object
    if type(function) == Expr_with_args:
        usenames = []
        for k in tmpargs:
            if params[k].vary == True:
                usenames.append(k)
        kws['Dfun'] = EWA_to_gradfunc(function, usenames)

    # if conversion function is specified, wrap func and Dfun
    if convert is not None:
        if callable(convert):
            convert_name = convert.__name__
            fullargs = ['x'] + tmpargs
            funcstr = """def wrapped(%s):
                ret = %s(func(%s))
                return ret""" % (' ,'.join(fullargs), 'convert', ', '.join(fullargs))
            Dfunstr = """def wrapped_Dfun(%s):
                ret = %s(Dfun(%s))
                return ret""" % (' ,'.join(fullargs), 'convert', ', '.join(fullargs))
            # make namespace for exec
            ns = globals()
            ns.update(locals())
            exec( funcstr, ns )
            if 'Dfun' in kws:
                # update kws['Dfun'] to wrapped one
                ns['Dfun'] = kws['Dfun']
                exec( Dfunstr, ns )
                kws['Dfun'] = wrapped_Dfun
            func = wrapped
        else:
            raise RuntimeError("can't call convert function %s" % convert)



    if 'Dfun' in kws:
        # https://github.com/lmfit/lmfit-py/blob/master/examples/example_derivfunc.py
        Dfun_orig = kws['Dfun']
        if err is None:
            def Dfun_pars(p, x, data=None):
                v = dict((k, v) for (k, v) in p.valuesdict().items() if k in tmpargs)
                ret = Dfun_orig(x, **v)
                return ret
        else:
            def Dfun_pars(p, x, data=None):
                v = dict((k, v) for (k, v) in p.valuesdict().items() if k in tmpargs)
                ret = Dfun_orig(x, **v)/err
                return ret
        kws['Dfun']      = Dfun_pars
        kws['col_deriv'] = 1

    # assert funcargs[0][0] == 'x'   # for now

    if err is None:
        # treat all error as 1
        def residue(p, xs):
            v = dict((k, v) for (k, v) in p.valuesdict().items() if k in tmpargs)
            return (func(xs, **v) - ys)
    else:
        # w/ err
        def residue(p, xs):
            v = dict((k, v) for (k, v) in p.valuesdict().items() if k in tmpargs)
            return (func(xs, **v) - ys)/err

    try:
        minimized = lmfit.minimize(residue, params, args=(xs,), **kws)
    except NameError as e: ## for EWA with un derivable expression
        if e.args != ("global name 'Derivative' is not defined",):
            raise
        else:
            ## retry without Dfun
            del kws["Dfun"]
            minimized = lmfit.minimize(residue, params, args=(xs,), **kws)

    minimized = lmfit.minimize(residue, params, args=(xs,), **kws)
    result    = FitResult(minimized, err, function)

    if not silent:
        result.report()

    return result
