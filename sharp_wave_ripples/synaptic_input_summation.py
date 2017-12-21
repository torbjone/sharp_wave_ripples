import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as ff

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


def return_threshold_crossing(sig):

    for t_idx in range(1, len(sig)):
        if sig[t_idx - 1] < 0.8 <= sig[t_idx]:
            return t_idx
    raise RuntimeError("Should not get here. No spike detected")

# np.random.seed(123456)

num_tsteps = 2**14
dt = 2**-5
syn_tau = 2
syn_length = 2**9
threshold = 0.8
num_trials = 500
num_spikes_list = np.array([1, 5, 25, 100, 500, 1000, 5000])

synaptic_input_SD = 1

tvec = np.arange(num_tsteps) * dt
center_idx = len(tvec) / 2

syn = np.zeros(syn_length)
syn[20:] = np.exp(-tvec[:syn_length - 20] / syn_tau)

tvec -= tvec[center_idx]

sigs = {}
threshold_crossings = {}
fig = plt.figure(figsize=[9,9])
fig.subplots_adjust(hspace=0.5)
ax1 = fig.add_subplot(221, title="Single synaptic input", xlabel="Time [ms]")
ax1.plot(tvec[:syn_length] - tvec[0], syn, 'k')
ax2 = fig.add_subplot(222, xlim=[-20, 20], xlabel="Time [ms]",
                      title="Normalized sum of normally \n distributed (SD: {} ms) synaptic inputs\n(One trial)".format(synaptic_input_SD))
ax2.axhline(threshold, linestyle=':', c='gray')

ax3 = fig.add_subplot(223, ylabel="ms", xlabel="Number of synaptic inputs", xscale="log",
                      title="Time at which signal reach\n80% of max value ({} trials)".format(num_trials))
ax4 = fig.add_subplot(224, title="SD of time of reaching 80 % of max value", ylabel="SD (ms)",
                      xlabel="Number of synaptic inputs")

lines = []
line_names = []
for num_spikes in num_spikes_list:

    for trial in range(num_trials):
        sig = np.zeros(num_tsteps)
        random_idxs = np.array(np.random.normal(center_idx, synaptic_input_SD/ dt, size=num_spikes), dtype=int)

        for syn_idx, idx in enumerate(random_idxs):

            t0 = idx
            t1 = idx + syn_length
            sig[t0:t1] += syn

        sig_n = sig / np.max(sig)
        if trial == 0:
            l1, = ax2.plot(tvec, sig_n)
            lines.append(l1)
            line_names.append("{} synaptic input".format(num_spikes))
            threshold_crossings[num_spikes] = []

        sigs[num_spikes] = sig_n

        spike_time_idx = return_threshold_crossing(sig_n)
        threshold_crossings[num_spikes].append(tvec[spike_time_idx])

stds = np.zeros(len(num_spikes_list))
for idx, num_spikes in enumerate(num_spikes_list):
    stds[idx] = np.std(threshold_crossings[num_spikes])
    ax3.scatter([num_spikes] * num_trials, threshold_crossings[num_spikes])

ax4.loglog(num_spikes_list, stds, 'kx-')

fig.legend(lines, line_names, frameon=False, ncol=4)



from plotting_convention import mark_subplots, simplify_axes
simplify_axes(fig.axes)
mark_subplots(fig.axes)
# fig.legend([l1, l2], ["'Correlated': {} * 1 spiketrain".format(num_synapses),
#                       "'Uncorrelated': 1 * {} spiketrains".format(num_synapses)], loc="lower center",
#            frameon=False, ncol=1)

plt.savefig("input_summation_4.png")


