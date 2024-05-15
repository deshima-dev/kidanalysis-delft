# -*- coding: utf-8 -*-
#### misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def down_sample(x, nsample):
    if hasattr(x, 'down_sample'):
        return x.down_sample(nsample)
    else:
        D = []
        for i in range(int(np.ceil(len(x)/float(nsample)))):
            beg = i*nsample
            end = min(len(x)-1, (i+1)*nsample)
            D.append(np.average(x[beg:end]))
        return np.array(D)

### exception class
class MKIDDataException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

### test functions
def guess_cabledelay(sd):
    """
    Guess cable delay v from SweepData
    """
    dx     = sd.x[1] - sd.x[0]
    dangle = sd.deg[1:] - sd.deg[:-1]
    return -np.average(dangle[abs(dangle)<180])*np.pi/180.0/dx

def plot_iq(iq, *args, **kwargs):
    plt.plot(np.real(iq), np.imag(iq), *args, **kwargs)

def get_dtheta(r, fx):
    theta_r  = np.angle(r.fitted(r.fitparams['fr'].value))
    theta_fx = np.average(np.angle(fx.iq))

    return theta_r - theta_fx

def patch_if(data, cond):
    indices = np.argwhere(cond)
    result = data[:]
    for i in indices:
        if i == 0:
            result[i] = result[i+1]
        elif i == len(data)-1:
            result[i] = result[i-1]
        else:
            result[i] = (data[i+1]+data[i-1])/2.0
    return result

class ListTable(list):
    """
    Overridden list class which takes a 2-dimensional list of the form [[1,2,3],[4,5,6]], and renders an HTML Table in IPython Notebook.
    """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")

            for col in row:
                html.append("<td>{0}</td>".format(col))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def summary(rs):
    res = ListTable()
    res.append( ('fr', 'Qr', 'Qi', 'phi0', 'WSSR') )
    import ad.admath as am
    for r in rs:
        c = ErrorCalculator(r.fitparams)
        Qi = 1/(1/c['Qr'] - 1/c['Qc']*am.cos(c['phi0']))
        res.append((c.str(c['fr']),c.str(c['Qr']),c.str(Qi),c.str(c['phi0']), r.info['s_sq']))
    return res

def rebin_log(x,y, begin=None, base=2, factor=2, average=True):
    """
    Rebin x, y (x is isospaced)

    :param begin: at what x rebin starts
    :param base: rebin bin width changes at begin*(base**n), n=0, 1, 2, ...
    :param factor: rebin bin width multiplier
    :param average: do average over new bin?
    """
    x_, y_ = [], []
    ind = 0
    width = 1
    if begin is None:
        nextx = 10**(np.floor(np.log(x[0])/np.log(10.0))+2)
    else:
        nextx = begin
    while ind < len(y):
        if x[ind] >= nextx:
            width *= factor
            nextx = nextx*base
        bintop = min(ind+width, len(y))
        binwidth = bintop-ind
        x_.append(x[ind])
        if average:
            y_.append(sum(y[ind:bintop]/float(binwidth)))
        else:
            y_.append(sum(y[ind:bintop]))
        ind += binwidth
    return np.array(x_), np.array(y_)
