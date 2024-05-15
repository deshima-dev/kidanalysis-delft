import numpy as np

def _numdif_2(y, dx):
    return (y[2:]+y[:-2]-2*y[1:-1])/dx**2


def _clusterize_indices(indices, threshold):
    """Fill small gap (within `threshold`) in indices.
    
    e.g.
     If threshold == 1,
      [True, False, False, True, True] => [True, False, False, True, True].
     If threshold == 2,
      [True, False, False, True, True] => [True, True, True, True, True].


    Parameters:
        indices:   an 1-D array of Boolean
        threshold: an interger, allowed gap between True's in indices"""

    results = np.copy(indices)
    prev  = 0
    first = True
    for i, x in enumerate(results):
        if x:
            if (not first) and i - prev <= threshold + 1:
                for j in range(prev, i):
                    results[j] = True
            prev  = i
            first = False
    return results


def deglitch(yss, sources=None,
             baseline_thresh = 6.0, glitch_thresh = 5.0, clusterize_thresh = 2,
             debug=False):
    """Deglitch `yss` using `sources`, assuming glitch exists at the same
    time of all data in `yss`. Too close glitches or broad glitch are
    treated as one glitch.
    
    Parameters:
        yss:               an array of 1-D arrays of data to deglitch
        sources:           source data to detect glitch. If None, yss is used
        baseline_thresh:   threshold to decide baseline
        glitch_thresh:     threshold to decide as glitch 
        clusterize_thresh: if gap between glitches are less than or equal to this,
                           treat them as one glitch.
        
        debug:   do plot for debugging
        
    Return:
        results: an array of 1-D arrays, that is deglitched yss"""
        
    if sources is None:
        sources = yss
    ave = np.average(sources, axis=0)
    xs  = np.arange(len(yss[0]))
    dx  = xs[1] - xs[0]
    diff2 = np.array(_numdif_2(ave, dx))
    sigma = np.std(diff2)
    good  = np.abs(diff2) < (baseline_thresh*sigma)
    sigma = np.std(diff2[good])
    bad   = (np.abs(diff2) >= glitch_thresh*sigma)
    ## treat broad glitch (or too close glitches) as one glitch
    bad   = _clusterize_indices(bad, clusterize_thresh)
    good  = np.logical_not(bad)
    results = []

    for ys in yss:
        center = np.interp(xs[1:-1], xs[1:-1][good], ys[1:-1][good])
        deglitched = np.concatenate(([ys[0]], center, [ys[-1]]))
        results.append(deglitched)
        
    if debug:
        import matplotlib.pyplot as plt
        import bokeh.plotting as bp
        import bokeh.mpl as bm
        f1 = plt.figure()
        plt.plot(xs, ave)
        plt.xlabel('data #')
        plt.ylabel('deglitched (green)')

        plt.plot(xs[1:-1][bad], ave[1:-1][bad], 'r+')
        for ys_deg, ys in zip(results, yss):
            plt.plot(xs, ys, 'y')
            plt.plot(xs, ys_deg, 'g')


        f2 = plt.figure()
        plt.plot(xs[1:-1], diff2)
        plt.xlabel('data #')
        plt.ylabel('2nd deriv')

        plt.plot(xs[1:-1][bad], diff2[bad], 'r+')

        s1 = bm.to_bokeh(f1)
        s2 = bm.to_bokeh(f2)
        s2.x_range = s1.x_range
        p = bp.gridplot([[s1, s2]])
        bp.show(p)
    return results

