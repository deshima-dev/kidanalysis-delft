import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#### guess error from sweep

def get_error_fixed_rp(data, cabledelay, verbose = False):
    """
    Get error from Nx256 observation data

    :param data: FixedData
    :param cabledelay: rad/gHz
    """
    if verbose:
        print( ':get_error:' )

    p = data
    L     = len(p)
    delta = L/256

    foo = p.down_sample(256).iq
    c = np.average(foo)
    a = np.angle(c)
    unrot = (foo-c)*np.exp(-1j*a)
    error_r = np.std(np.real(unrot))
    error_p = np.std(np.imag(unrot))/abs(c)
    error_f = error_p / cabledelay
    # print( error_r, error_f )
    if verbose:
        pass
    return (error_r, error_f)

def get_error_fixed_iq(p, verbose = False):
    """
    Get error from Nx256 observation data
    """
    if verbose:
        print( ':get_error:' )

    L     = len(p)
    delta = L/256

    foo = p.down_sample(256).iq
    c = np.average(foo)
    a = np.angle(c)
    residue = foo-c
    error_i = np.std(np.real(residue))
    error_q = np.std(np.imag(residue))
    return (error_i, error_q)


def get_error_sweep_rp(data, message=False, plot=False):
    """
    Estimate error by fitting iq data as 2d-vector data, with linear function
    """
    def clinear_fitfunc(x, a, b):
        return a*x + b
    def clinear_residual(param, x, y):
        return y - clinear_fitfunc(x,*param)

    # initial value
    x = data.x
    y = data.db
    iq = data.iq
    # a = (y[-1]-y[0])/(x[-1]-x[0])
    # b = y[0] - a*x[0]

    slope_r, intercept_r, _, _, _ = scipy.stats.linregress(x, np.real(iq))
    slope_i, intercept_i, _, _, _ = scipy.stats.linregress(x, np.imag(iq))
    slope     = slope_r     + 1j * slope_i
    intercept = intercept_r + 1j * intercept_i
    res = [slope, intercept]

    residual_ = clinear_residual(res, x, iq)
    residual  = residual_ * np.exp(-1j*np.angle(slope))

    R     = abs(np.average(iq))
    dfdx  = (x[1] - x[0])
    # v     = abs(slope) / R / dfdx
    # print( 'v_', v )
    ndf   = (len(data)) - 2
    s_sq_radius    = (np.imag(residual)**2).sum()/ndf
    s_sq_frequency = ((np.real(residual)/R)**2).sum()/ndf
    noise_r = np.sqrt(s_sq_radius)
    noise_f = np.sqrt(s_sq_frequency)
    print( noise_r, noise_f*R )

    if(plot):
        y1 = np.real(iq)
        y2 = np.imag(iq)
    
        fity  = clinear_fitfunc(x, *res)
        fig   = plt.figure()
        ax    = fig.add_subplot(111)
        ax.plot(x,data.amplitude,'-g', label='Amplitude')
        ax.plot(x,y1,'-b', label='I Axis(real)')
        ax.plot(x,y2,'-r', label='Q Axis(imag)')
        ax.plot(x,abs(fity),'-y', label='Amplitude(fit)')
        ax.plot(x,np.real(fity),'-c', label='I Axis(fit)')
        ax.plot(x,np.imag(fity),'-m', label='Q Axis(fit)')
        ax.legend()
        ax.grid()

        fig   = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        # ax = fig.add_subplot(111)
        ax.plot(y1,y2)
        ax.plot(np.real(fity), np.imag(fity), '-r')

        ax.plot(data.i, data.q, '-b', label='data')
        ax.plot(np.real(fity), np.imag(fity), '-r', label='fit')
        ax.set_title('linear fit for error estimation')
        plt.show()

        fity  = clinear_fitfunc(x, *res)
        fig   = plt.figure()

        ax    = fig.add_subplot(111, aspect='equal')
        ax.plot(np.real(residual), np.imag(residual))
        ax.grid()
        plt.show()


        fig   = plt.figure()
        ax    = fig.add_subplot(111, aspect='equal')
        # ax = fig.add_subplot(111)
        

        ax.plot(data.i - np.real(fity), data.q - np.imag(fity), '*b', label='residual')
        ax.set_title('linear fit residual')
        plt.show()
    return noise_r, noise_f

def get_error_sweep_iq(data, message=False, plot=False):
    """
    Estimate error by fitting iq data as 2d-vector data, with linear function
    """
    def clinear_fitfunc(x, a, b):
        return a*x + b
    def clinear_residual(param, x, y):
        return y - clinear_fitfunc(x,*param)

    # initial value
    x = data.x
    y = data.db
    iq = data.iq
    a = (y[-1]-y[0])/(x[-1]-x[0])
    b = y[0] - a*x[0]

    slope_r, intercept_r, _, _, _ = scipy.stats.linregress(x, np.real(iq))
    slope_i, intercept_i, _, _, _ = scipy.stats.linregress(x, np.imag(iq))
    slope     = slope_r     + 1j * slope_i
    intercept = intercept_r + 1j * intercept_i
    res = [slope, intercept]

    residual = clinear_residual(res, x, iq)
    ndf   = (len(data)) - 2
    s_sq_r  = (np.real(residual)**2).sum()/ndf
    s_sq_i  = (np.imag(residual)**2).sum()/ndf

    if(plot):
        y1 = np.real(iq)
        y2 = np.imag(iq)
    
        fity  = clinear_fitfunc(x, *res)
        fig   = plt.figure()
        ax    = fig.add_subplot(111)
        ax.plot(x,data.amplitude,'-g', label='Amplitude')
        ax.plot(x,y1,'-b', label='I Axis(real)')
        ax.plot(x,y2,'-r', label='Q Axis(imag)')
        ax.plot(x,abs(fity),'-y', label='Amplitude(fit)')
        ax.plot(x,np.real(fity),'-c', label='I Axis(fit)')
        ax.plot(x,np.imag(fity),'-m', label='Q Axis(fit)')
        ax.legend()
        ax.grid()

        fig   = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        # ax = fig.add_subplot(111)
        ax.plot(y1,y2)
        ax.plot(np.real(fity), np.imag(fity), '-r')

        ax.plot(data.i, data.q, '-b', label='data')
        ax.plot(np.real(fity), np.imag(fity), '-r', label='fit')
        ax.set_title('linear fit for error estimation')
        plt.show()

        fig   = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        # ax = fig.add_subplot(111)

        ax.plot(data.i - np.real(fity), data.q - np.imag(fity), '*b', label='residual')
        ax.set_title('linear fit residual')
        plt.show()
    return np.sqrt(s_sq_r), np.sqrt(s_sq_i)

