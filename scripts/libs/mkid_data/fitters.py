import collections
import sympy
import numpy as np
from fit.expr_with_args import Expr_with_args
from .peak_search import search_peak, search_peaks


Fitter = collections.namedtuple('Fitter', 'func, guess, paramnames, info')

complex_fitters = ['mazinrev', 'gao', 'gaolinbg', 'blank', 'gao2']
real_fitters    = []
all_fitters     = complex_fitters + real_fitters

#################################################################
# mazinrev: a complex function from Mazin's D thesis
################################################################
mazinrev_paramnames = ['FWHM', 'f0', 'a_on', 'a_off', 'v', 'c', 'theta', 'gr', 'Ic', 'Qc']
mazinrev_paramlatex = r'FWHM, f_0 a_{on} a_{off} v c theta g_r I_c Q_c'
def make_EWAmazinrev():
    arg_names = mazinrev_paramnames
    arg_latex = mazinrev_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    FWHM, f0, a_on, a_off, v, c, theta, gr, Ic, Qc = arg_symbols

    from sympy import exp, I, pi, re, im

    x = sympy.symbols('x')
    deltax = x - f0
    origx = f0
    w = deltax/FWHM
    f = a_off + (a_off - a_on)*(2*I*w/(1+2*I*w) - 1) + c*deltax
    expr = (re(f)+I*gr*im(f))*exp(I*(theta-v*deltax)) + (Ic + I*Qc)
    return Expr_with_args(expr, arg_symbols, arg_names)

def make_EWAmazinrev_bg():
    arg_latex = mazinrev_paramlatex
    arg_names = mazinrev_paramnames

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    FWHM, f0, a_on, a_off, v, c, theta, gr, Ic, Qc = arg_symbols

    from sympy import exp, I, pi, re, im

    x = sympy.symbols('x')
    deltax = x - f0
    origx = f0
    w = deltax/FWHM
    f = a_off + c*deltax
    expr = (re(f)+I*gr*im(f))*exp(I*(theta-v*deltax)) + (Ic + I*Qc)
    return Expr_with_args(expr, arg_symbols, arg_names)

def mazinrev_guess(data):
    x      = data.x
    deltax = x[1] - x[0]
    pdict   = search_peak(data)
    y0      = data.iq[pdict['f0ind']]
    FWHM    = pdict['f0']/pdict['Q']
    ddeg    = data.deg[1:] - data.deg[:-1]
    v       = -np.average(ddeg[abs(ddeg)<180])*np.pi/180.0/deltax
    theta   = np.angle(y0)
    gr      = 1.0
    y_unrev = data.iq * np.exp(-1j*(-v*(x-pdict['f0']) + theta))
    c       = 0.0
    Ic, Qc  = 0.0, 0.0
    return dict(zip(mazinrev_paramnames, (FWHM, pdict['f0'], pdict['a_on'], pdict['a_off'], v, c, theta, gr, Ic, Qc)))

def mazinrev_rewind(x, y, FWHM, f0, a_on, a_off, v, c, theta, gr, Ic, Qc):
    deltax = x - f0
    f2  = (y - (Ic + 1j*Qc))/np.exp(1j*(theta-v*deltax))
    f   = np.real(f2) + 1j*np.imag(f2)/gr
    y_  = f - c*deltax
    return (y_ - a_off)/(a_off - a_on) + 0.5

mazinrev_ewa    = make_EWAmazinrev()
mazinrev_ewa_bg = make_EWAmazinrev_bg()
fitter_mazinrev = Fitter(mazinrev_ewa, mazinrev_guess, mazinrev_paramnames,
                         {'bgfunc': mazinrev_ewa_bg, 'rewindfunc': mazinrev_rewind})

#################################################################
# gaolinbg: a complex function from Gao's D thesis, plus linear term
################################################################
gaolinbg_paramnames = 'arga absa tau fr Qr Qc phi0 c'.split()
gaolinbg_paramlatex = r'(\arg{a}) |a| tau f_r Q_r Q_c phi_0 c'
def make_EWAgaolinbg():
    arg_names = gaolinbg_paramnames
    arg_latex = gaolinbg_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    arga, absa, tau, fr, Qr, Qc, phi0, c = arg_symbols

    from sympy import exp, I, pi

    x = sympy.symbols('x')
    expr = (absa * exp(-I*(2*pi*x*tau - arga))*
            (1+c*(x-fr)-Qr/Qc*exp(I*phi0)/(1+2*I*Qr*((x-fr)/fr))))

    return Expr_with_args(expr, arg_symbols, arg_names)

def make_EWAgaolinbg_bg():
    arg_names = gaolinbg_paramnames
    arg_latex = gaolinbg_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    arga, absa, tau, fr, Qr, Qc, phi0, c = arg_symbols

    from sympy import exp, I, pi

    x = sympy.symbols('x')
    expr = (absa * exp(-I*(2*pi*x*tau - arga))*
            (1+c*(x-fr)))

    return Expr_with_args(expr, arg_symbols, arg_names)

def gaolinbg_guess(data):
    x       = data.x
    deltax  = x[1] - x[0]
    pdict   = search_peak(data)
    y0      = data.iq[pdict['f0ind']]
    FWHM    = pdict['f0']/pdict['Q']
    ddeg    = data.deg[1:] - data.deg[:-1]
    tau     = -np.average(ddeg[abs(ddeg)<180])*np.pi/180.0/deltax/2/np.pi
    f0      = pdict['f0']
    theta   = np.angle(y0)
    arga    = np.angle(y0*np.exp(1j*2*np.pi*tau*f0))
    absa    = pdict['a_off']
    fr      = f0
    Qr      = f0/FWHM
    Qc      = Qr
    phi0    = 0
    c       = 0
    
    return dict(zip(gaolinbg_paramnames, (arga, absa, tau, fr, Qr, Qc, phi0, c)))

def gaolinbg_rewind(x, y, arga, absa, tau, fr, Qr, Qc, phi0, c):
    tmp = y/absa/np.exp(-1j*(2*np.pi*x*tau-arga)) - c*(x-fr)
    return (tmp-1)*Qc/Qr/np.exp(1j*phi0) + 0.5

gaolinbg_ewa    = make_EWAgaolinbg()
gaolinbg_bgewa  = make_EWAgaolinbg_bg()
fitter_gaolinbg = Fitter(gaolinbg_ewa, gaolinbg_guess, gaolinbg_paramnames,
                         {'bgfunc': gaolinbg_bgewa, 'rewindfunc': gaolinbg_rewind,
                          'additional_expr': {'Qi': '1/(1/Qr - 1/Qc*cos(phi0))'},
                          # 'prefit_and_fix': []
                         })

#################################################################
# gao: a complex function from Gao's D thesis
################################################################
gao_paramnames = 'arga absa tau fr Qr Qc phi0'.split()
gao_paramlatex = r'(\arg{a}) |a| tau f_r Q_r Q_c phi_0'

def make_EWAgao():
    arg_names = gao_paramnames
    arg_latex = gao_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    arga, absa, tau, fr, Qr, Qc, phi0 = arg_symbols

    from sympy import exp, I, pi

    x = sympy.symbols('x')
    expr = (absa * exp(-I*(2*pi*x*tau - arga))*
            (1-Qr/Qc*exp(I*phi0)/(1+2*I*Qr*((x-fr)/fr))))

    return Expr_with_args(expr, arg_symbols, arg_names)

def make_EWAgao_bg():
    arg_names = gao_paramnames
    arg_latex = gao_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    arga, absa, tau, fr, Qr, Qc, phi0 = arg_symbols

    from sympy import exp, I, pi

    x = sympy.symbols('x')
    expr = (absa * exp(-I*(2*pi*x*tau - arga)))

    return Expr_with_args(expr, arg_symbols, arg_names)

def gao_guess(data):
    x      = data.x
    deltax = x[1] - x[0]
    pdict   = search_peak(data)
    y0      = data.iq[pdict['f0ind']]
    FWHM    = pdict['f0']/pdict['Q']
    ddeg    = data.deg[1:] - data.deg[:-1]
    tau     = -np.average(ddeg[abs(ddeg)<180])*np.pi/180.0/deltax/2/np.pi
    f0      = pdict['f0']
    theta   = np.angle(y0)
    arga    = np.angle(y0*np.exp(1j*2*np.pi*tau*f0))
    absa    = pdict['a_off']
    fr      = f0
    Qr      = f0/FWHM
    Qc      = Qr
    phi0    = 0
    
    return dict(zip(gao_paramnames, (arga, absa, tau, fr, Qr, Qc, phi0)))

def gao_rewind(x, y, arga, absa, tau, fr, Qr, Qc, phi0):
    tmp = y/absa/np.exp(-1j*(2*np.pi*x*tau-arga))
    return (tmp-1)*Qc/Qr + 0.5

gao_ewa    = make_EWAgao()
gao_bgewa  = make_EWAgao_bg()
fitter_gao = Fitter(gao_ewa, gao_guess, gao_paramnames,
                    {'bgfunc': gao_bgewa, 'rewindfunc': gao_rewind,
                     'additional_expr': {'Qi': '1/(1/Qr - 1/Qc*cos(phi0))'},
                     'prefit_and_fix': []
                    })

################################################################
# blank: a complex function to fit baseline for cable delay
################################################################
blank_paramnames = ['x0', 'theta0', 'a_off', 'v', 'gr', 'Ic', 'Qc']
blank_paramlatex = ['x0', 'theta_0', 'a_{off}', 'v', 'g_r', 'I_c', 'Q_c']

def make_EWAblank():
    arg_names = blank_paramnames
    arg_latex = blank_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    x0, theta0, a_off, v, gr, Ic, Qc= arg_symbols

    from sympy import exp, I, pi, re, im

    x = sympy.symbols('x')

    deltax = x - x0
    f      = -a_off*exp(I*(-v*deltax))
    expr   = (re(f)+I*gr*im(f))*exp(I*theta0) + (Ic + I*Qc)

    return Expr_with_args(expr, arg_symbols, arg_names)

def blank_guess(data):
    x          = data.x
    y          = data.iq
    deltax     = x[1] - x[0]
    deltaangle = data.deg[1:]-data.deg[:-1]

    v      = -np.average(deltaangle[abs(deltaangle)<180])*np.pi/180.0/deltax
    x0     = x[0]
    theta0 = np.angle(y[0]) - np.pi
    a_off  = np.median(abs(y))
    Ic     = 0.0
    Qc     = 0.0
    gr     = 1.0

    return dict(zip(blank_paramnames, [x0, theta0, a_off, v, gr, Ic, Qc]))

blank_ewa    = make_EWAblank()
fitter_blank = Fitter(blank_ewa, blank_guess, blank_paramnames,
                      {})


################################################################
# gao2: a complex function for 2 KIDs simultaneous fitting
################################################################

gao2_paramnames = 'arga absa tau fr Qr Qc phi0 c fr_1 Qr_1 Qc_1 phi0_1'.split()
gao2_paramlatex = r'(\arg{a}) |a| tau f_r Q_r Q_c phi_0 c f_{r1} Q_{r1} Q_{c1} phi_{01}'
def make_EWAgao2():
    arg_names = gao2_paramnames
    arg_latex = gao2_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    arga, absa, tau, fr, Qr, Qc, phi0, c, fr_1, Qr_1, Qc_1, phi0_1 = arg_symbols

    from sympy import exp, I, pi

    x = sympy.symbols('x')
    expr = (absa * exp(-I*(2*pi*x*tau - arga))*
            (1+c*(x-fr)-Qr/Qc*exp(I*phi0)/(1+2*I*Qr*((x-fr)/fr)) + 
                       -Qr_1/Qc_1*exp(I*phi0_1)/(1+2*I*Qr_1*((x-fr_1)/fr_1)))
            )

    return Expr_with_args(expr, arg_symbols, arg_names)

def make_EWAgao2_bg():
    arg_names = gao2_paramnames
    arg_latex = gao2_paramlatex

    ## define symbols to use as parameters
    arg_symbols = sympy.symbols(arg_latex)
    arga, absa, tau, fr, Qr, Qc, phi0, c, fr_1, Qr_1, Qc_1, phi0_1 = arg_symbols

    from sympy import exp, I, pi

    x = sympy.symbols('x')
    expr = (absa * exp(-I*(2*pi*x*tau - arga))*
            (1+c*(x-fr)))

    return Expr_with_args(expr, arg_symbols, arg_names)

def gao2_guess(data, Q_search=1000):
    x       = data.x
    deltax  = x[1] - x[0]
    pd      = search_peaks(data, fc=None, Q_search=Q_search)
    pdict   = pd[0]
    y0      = data.iq[pdict['f0ind']]
    FWHM    = pdict['f0']/pdict['Q']
    ddeg    = data.deg[1:] - data.deg[:-1]
    tau     = -np.average(ddeg[abs(ddeg)<180])*np.pi/180.0/deltax/2/np.pi
    f0      = pdict['f0']
    theta   = np.angle(y0)
    arga    = np.angle(y0*np.exp(1j*2*np.pi*tau*f0))
    absa    = pdict['a_off']
    fr      = f0
    Qr      = f0/FWHM
    Qc      = Qr
    phi0    = 0
    c       = 0

    pdict_1   = pd[1]
    y0_1      = data.iq[pdict_1['f0ind']]
    FWHM_1    = pdict_1['f0']/pdict_1['Q']
    f0_1      = pdict_1['f0']
    theta_1   = np.angle(y0_1)
    fr_1      = f0_1
    Qr_1      = f0_1/FWHM_1
    Qc_1      = Qr_1
    phi0_1    = 0
    
    return dict(zip(gao2_paramnames, (arga, absa, tau, fr, Qr, Qc, phi0, c, 
                                      fr_1, Qr_1, Qc_1, phi0_1)))

def gao2_rewind(x, y, arga, absa, tau, fr, Qr, Qc, phi0, c,
                fr_1, Qr_1, Qc_1, phi0_1):
    tmp = y/absa/np.exp(-1j*(2*np.pi*x*tau-arga)) - c*(x-fr) + Qr_1/Qc_1*np.exp(1j*phi0_1)/(1+2*1j*Qr_1*((x-fr_1)/fr_1))
    return (tmp-1)*Qc/Qr/np.exp(1j*phi0) + 0.5

gao2_ewa    = make_EWAgao2()
gao2_bgewa  = make_EWAgao2_bg()
fitter_gao2 = Fitter(gao2_ewa, gao2_guess, gao2_paramnames,
                     {'bgfunc': gao2_bgewa, 'rewindfunc': gao2_rewind,
                      'additional_expr': {'Qi': '1/(1/Qr - 1/Qc*cos(phi0))',
                                          'Qi_1': '1/(1/Qr_1 - 1/Qc_1*cos(phi0_1))'},
                      # 'prefit_and_fix': []
                      })
