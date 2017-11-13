import numpy as np
import scipy.fftpack as ff
import scipy.signal as ss
import pylab as plt
# NEURON is not OK with this package for some reason. Seems to be something about __future__ import related to strings
from matplotlib import mlab as ml

def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]


    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power

def return_freq_and_psd_and_phase(tvec, sig):
    """ Returns the power and freqency of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    phase = np.angle(Y)
    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power, phase

def return_freq_and_psd_welch(sig, welch_dict):
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    psd = []
    freqs = None
    for idx in xrange(sig.shape[0]):
        yvec_w, freqs = ml.psd(sig[idx, :], **welch_dict)
        psd.append(yvec_w)
    return freqs, np.array(psd)


def filter_data(dt, data, low_freq=20.0, high_freq=3000.):
     data_filter = {
            'filter_design' : ss.ellip,
            'filter_design_args' : {
                'N' : 2,
                'rp': 0.1,
                'rs': 40,
                'Wn' : np.array([low_freq, high_freq]) * dt / 1000. *2,
                'btype' : 'bandpass',
            },
            'filter' : ss.filtfilt
     }
     b, a = data_filter['filter_design'](**data_filter['filter_design_args'])
     data_filtered = data_filter['filter'](b, a, data)
     return data_filtered


def test_filter():

    dt = 2**-3
    tvec = np.arange(2**15) * dt
    sig = np.random.normal(0, 1., size=len(tvec))
    filt_sig = filter_data(dt, sig)

    plt.subplot(121)
    plt.plot(tvec, sig, 'k')
    plt.plot(tvec, filt_sig, 'r')

    freq, psd = return_freq_and_psd(tvec, sig)
    freq, filt_psd = return_freq_and_psd(tvec, filt_sig)

    plt.subplot(122)

    plt.loglog(freq, psd[0], 'k')
    plt.loglog(freq, filt_psd[0], 'r')

    plt.show()


if __name__ == '__main__':
    test_filter()