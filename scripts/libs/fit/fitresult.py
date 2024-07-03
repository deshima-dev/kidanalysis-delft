import collections
import inspect
import json
import pickle
from copy import deepcopy
#import ad
import lmfit
import scipy.stats
import numpy as np

class FitResult(object):
    """Class to represent fit result, to be human-readble-output friendly""" 
    def __init__(self, minimized=None, error_given=None, function=None):
        if minimized is None and error_given is None and function is None:
            return

        if error_given is not None:
            self.error_is_asymptotic = False
        else:
            self.error_is_asymptotic = True

        if minimized.covar is not None:
            self.covar = minimized.covar
        else:
            self.covar = np.full([minimized.nvarys] * 2, np.nan)

        self.function = function
        self.init_vals = minimized.init_vals
        self.params = minimized.params
        self.info   = dict((k, getattr(minimized, k)) for k in
                           # json-friendly data taken from
                           # http://lmfit.github.io/lmfit-py/fitting.html#MinimizerResult
                           ['var_names',
                            'nfev',
                            'success',
                            'errorbars',
                            'message',
                            'ier',
                            'lmdif_message',
                            'nvarys',
                            'ndata',
                            'nfree',
                            'chisqr',
                            'redchi',
                            'aic',
                            'bic',])
        # self.p_value = scipy.stats.chi2.sf(self.info['chisqr'], self.info['nfree'])
    def report(self):
        print (self.report_str())

    def report_str(self):
        # taken from lmfit's printfuncs.py
        output = ""
        output += ("[[Fit Statistics]]\n")
        output += ("    # function evals   = %s\n" % self.info['nfev'])
        output += ("    # data points      = %s\n" % self.info['ndata'])
        output += ("    # variables        = %s\n" % self.info['nvarys'])
        output += ("    chi-square         = %s\n" % self.info['chisqr'])
        output += ("    reduced chi-square = %s\n" % self.info['redchi'])
        if not self.error_is_asymptotic:
            output += ('    p-value            = %s\n' %
                       scipy.stats.chi2.sf(self.info['chisqr'], self.info['nfree']))
        output += lmfit.fit_report(self.params)
        return output

    def values(self, with_expr_param=False):
        if with_expr_param:
            return dict([(k, v.value) for (k, v) in self.params.items()])
        else:
            return dict([(k, v.value) for (k, v) in self.params.items() if not v.expr])

    def errors(self, with_expr_param=False, prefix='d'):
        if with_expr_param:
            return dict([(prefix + k, v.stderr) for (k, v) in self.params.items()])
        else:
            return dict([(prefix + k, v.stderr) for (k, v) in self.params.items() if not v.expr])


    def eval(self, x):
        return self.function(x, **self.values())

    def dumps(self, **kws):
        out = dict()
        out['init_vals'] = self.init_vals
        out['params']    = json.loads(self.params.dumps(**kws))
        out['covar']     = self.covar.tolist()
        out['error_is_asymptotic'] = self.error_is_asymptotic
        # avoid "True is not json serializable", where True is numpy.bool_
        out['info']      = dict((k, bool(v) if type(v) == np.bool_ else v)
                            for (k, v) in self.info.items())
        out['function']  = pickle.dumps(self.function, 2).decode('latin-1')

        #print( out['function'].decode('base64') )
        return json.dumps(out, **kws)

    @classmethod
    def loads(cls, s, **kws):
        r = FitResult()
        dic = json.loads(s, **kws)
        for k, v in dic.items():
            if k == 'init_vals':
                r.init_vals = v
            elif k == 'function':
                #r.function = pickle.loads(v.encode('ascii'))
                r.function = pickle.loads(v.encode('latin-1'))
            elif k == 'params':
                r.params = lmfit.Parameters()
                r.params.loads(json.dumps(v))
            elif k == 'covar':
                r.covar = np.asarray(v)
            elif k == 'error_is_asymptotic':
                r.error_is_asymptotic = v
            elif k == 'info':
                r.info = v
            else:
                raise RuntimeError('unknown key: %s' % k)

        return r

    def dump(self, fp, **kws):
        return fp.write(self.dumps(**kws))
    @classmethod
    def load(cls, fp, **kws):
        return cls.loads(fp.read(), **kws)

    # avoid pickling problem of lmfit.Parameters
    def __getstate__(self):
        dic = self.__dict__.copy()
        if isinstance(dic['params'], lmfit.Parameters):
            dic['params'] = dic['params'].dumps()
        return dic
    def __setstate__(self, dic):
        if dic['params']:
            p = lmfit.Parameters()
            p.loads(dic['params'])
            dic['params'] = p
        self.__dict__ = dic
