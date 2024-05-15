# -*- coding: utf-8 -*-
#### psd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack
import sys
import warnings

#def power_spectrum_density(data, dt, ndivide=1, doplot=False, window=scipy.signal.hanning, overwrap_half=False):
def power_spectrum_density(data, dt, ndivide=1, doplot=False, window=np.hanning, overwrap_half=False):
    """
    Calculate power spectrum density of data.

    :param data:    input data
    :param dt:      time between each data
    :param ndivide: do averaging(split data into ndivide, get psd of each, and average them)
    :param doplot:  plot how averaging works
    :param overwrap_half:  a boolean, split data to half-overwrapped regions

    :return: (frequencies, psd)
    """
    if overwrap_half:
        step = int( len(data)/(ndivide+1) )
        size = step*2
    else:
        step = int( len(data)/ndivide )
        size = step
    # if bin(len(data)).count("1") != 1:
    #     # print( 'warning: length of data is not power of 2:', len(data), file=sys.stderr )
    #     warnings.warn('warning: length of data is not power of 2: %d' % len(data))
    # size = len(data)/ndivide
    if bin(size).count("1") != 1:
        # print( 'warning: (length of data)/ndivide is not power of 2:', size, file=sys.stderr )
        if overwrap_half:
            warnings.warn('warning: ((length of data)/(ndivide+1))*2 is not power of 2: %d' % size)
        else:
            warnings.warn('warning: (length of data)/ndivide is not power of 2: %d' % size)
    psd = np.zeros(size)
    T     = (size-1)*dt
    vs    = 1/dt
    vk_ = scipy.fftpack.fftfreq(size, dt)
    vk = vk_[np.where(vk_>=0)]
    # print( 'len(data), ndivide, dt, size, T, vs', len(data), ndivide, dt, size, T, vs )
    for i in range(ndivide):
        d = data[i*step:i*step+size]
        if window is None:
            w    = np.ones(size)
            corr = 1.0
        else:
            w    = window(size)
            corr = np.mean(w**2)
        psd  = psd + 2*(np.abs(scipy.fftpack.fft(d * w)))**2/size * dt / corr
        if doplot:
            if i == 0:
                print( 'initial' )
                plt.loglog(vk, psd[:len(vk)])
                plt.loglog(vk, 0*vk+1, '+')
                plt.xlabel('Frquency')
                plt.ylabel('psd')
                plt.show()
            elif i==1:
                print( 'averaging 2' )
                plt.loglog(vk, psd[:len(vk)]/2)
                plt.loglog(vk, 0*vk+1, '+')
                plt.xlabel('Frquency')
                plt.ylabel('psd')
                plt.show()
            elif i==ndivide-1:
                print( 'averaging', ndivide )
                plt.loglog(vk, psd[:len(vk)]/ndivide)
                plt.loglog(vk, 0*vk+1, '+')
                plt.xlabel('Frquency')
                plt.ylabel('psd')
                plt.show()
            plt.loglog(vk, psd[:len(vk)]/float(i+1))
    return vk, psd[:len(vk)]/ndivide

def cross_power_spectrum_density(data1, data2, dt, ndivide=1, doplot=False):
    """
    Calculate power spectrum density of data.

    :param data:    input data
    :param dt:      time between each data
    :param ndivide: do averaging(split data into ndivide, get psd of each, and average them)
    :param doplot:  plot how averaging works

    :return: (frequencies, psd)
    """
    if bin(len(data1)).count("1") != 1:
        print( 'warning: length of data is not power of 2:', len(data1), file=sys.stderr )
    size = int( len(data1)/ndivide )
    if bin(size).count("1") != 1:
        print( 'warning: (length of data)/ndivide is not power of 2:', size, file=sys.stderr )
    psd = np.zeros(size)
    T     = (size-1)*dt
    vs    = 1/dt
    vk_ = scipy.fftpack.fftfreq(size, dt)
    vk = vk_[np.where(vk_>=0)]
    # print( 'len(data), ndivide, dt, size, T, vs', len(data), ndivide, dt, size, T, vs )
    for i in range(ndivide):
        d1 = data1[i*size:(i+1)*size]
        d2 = data2[i*size:(i+1)*size]
        #w    = scipy.signal.hanning(size)
        w    = np.hanning(size)
        # psd  = psd + 2*(np.abs((scipy.fftpack.fft(d1 * w))*np.conj(scipy.fftpack.fft(d2 * w))))/size * dt / 0.3726
        psd  = psd + 2*(scipy.fftpack.fft(d1 * w)*np.conj(scipy.fftpack.fft(d2 * w)))/size * dt / 0.3726
        if doplot:
            if i == 0:
                print( 'initial' )
                plt.loglog(vk, psd[:len(vk)])
                plt.loglog(vk, 0*vk+1, '+')
                plt.xlabel('Frquency')
                plt.ylabel('psd')
                plt.show()
            elif i==1:
                print( 'averaging 2' )
                plt.loglog(vk, psd[:len(vk)]/2)
                plt.loglog(vk, 0*vk+1, '+')
                plt.xlabel('Frquency')
                plt.ylabel('psd')
                plt.show()
            elif i==ndivide-1:
                print( 'averaging', ndivide )
                plt.loglog(vk, psd[:len(vk)]/ndivide)
                plt.loglog(vk, 0*vk+1, '+')
                plt.xlabel('Frquency')
                plt.ylabel('psd')
                plt.show()
            plt.loglog(vk, psd[:len(vk)]/float(i+1))
    return vk, psd[:len(vk)]/ndivide
