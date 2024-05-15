#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt

__version__ = '0.8.1.1'

## to plot huge data
plt.rcParams['agg.path.chunksize'] = 10000

from .misc import *
from .data import *
# from .fitting import *
from .get_error import get_error_sweep_iq, get_error_fixed_iq, get_error_sweep_rp, get_error_fixed_rp
from .peak_search import find_peaks, search_peaks, search_peak, fitLorentzian
from .peak_search import deglitch, find_glitch, interpolate_bad
from .peak_search import find_glitch_advanced, tod_filt, find_glitch_both
from .peak_search import open_indices, close_indices, count_indices_cluster, indices_to_slices
from .fitters import fitter_gaolinbg, fitter_gao, fitter_mazinrev, fitter_blank
from .kidfit import fit_from_params, fit_onepeak, adjust_fitrange, KidFitResult


# from .errorCalculator import *
from .psd import *
# from .temperature import *

