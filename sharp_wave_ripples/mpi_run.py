import os
import sys
import numpy as np
from os.path import join
import matplotlib
matplotlib.use("AGG")
import pylab as plt
import tools
from plotting_convention import simplify_axes
from cell_simulation import return_cell, return_electrode_parameters, random_seed, random_seed_shift
import LFPy


root_folder = ".."


def prope_EAP_synchronizations(cell_name, num_cells):

    num_trials = 100
    plot_trials = True
    jitter_STDs = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])

    use_elec_idx = 10 if cell_name is "hbp_L4_SBC_bNAC219_1" else 7
    print use_elec_idx

    cell = return_cell(cell_name, 0)
    dt = cell.dt
    sim_name = 'swr_%s_%04d' % (cell_name, 0)
    print cell_name
    EAP = np.load(join(root_folder, cell_name, "EAPs", "EAP_%s.npy" % sim_name))

    composed_signal_length = 400
    composed_tvec = np.arange(0, composed_signal_length / dt + 1) * dt

    composed_center = 200
    composed_center_idx = np.argmin(np.abs(composed_tvec - composed_center))

    mean_latency = 10

    t = composed_tvec - composed_center + mean_latency

    signal_list = ["raw", "wband", "spikes", "lfp", "lfp_hf"]
    signal_filter_params = {"raw": None,
                            "wband": [20.0, 3000.],  # [low_freq, high_freq]
                            "spikes": [600., 3000.],
                            "lfp": [20., 600.],
                            "lfp_hf": [200., 600.],
                            }
    title_dict = {"raw": "Raw data",
                  "wband": "Wide band (20-3000 Hz)",
                  "spikes": "Spike (600-3000 Hz)",
                  "lfp": "LFP (20-600 Hz)",
                  "lfp_hf": "hf_LFP (200-600 Hz)",
                  }

    num_cols = len(jitter_STDs)
    latencies = {key: np.zeros((len(jitter_STDs), num_trials)) for key in signal_list}

    clr_dict = {"raw": "gray",
                "wband": "k",
                "spikes": "g",
                "lfp": "b",
                "lfp_hf": "c",
                }

    for trial_num in range(num_trials):
        print trial_num

        np.random.seed(num_cells + 1234 + trial_num)
        summed_EAP_dict = {j: np.zeros((EAP.shape[0], len(composed_tvec)))
                           for j in jitter_STDs}

        for cell_idx in range(0, num_cells):
            if cell_idx < 50:
                continue
            sim_name = 'swr_%s_%04d' % (cell_name, cell_idx)
            EAP = np.load(join(root_folder, cell_name, "EAPs", "EAP_%s.npy" % sim_name))

            for jit_idx, jitter_STD in enumerate(jitter_STDs):
                jitter_idx = int(round(np.random.normal(0, jitter_STD / cell.dt)))
                t0 = composed_center_idx + jitter_idx - EAP.shape[1] / 2
                t1 = t0 + EAP.shape[1]
                summed_EAP_dict[jitter_STD][:, t0:t1] += EAP[:, :]

        if plot_trials:
            plt.close("all")
            fig = plt.figure(figsize=[18, 9])
            fig.subplots_adjust(wspace=0.5, right=0.98, left=0.1, hspace=0.5)
            max_sig = np.max(np.abs(summed_EAP_dict[0][use_elec_idx])) / 1000
            xlim = [5, 15]
            ylim = [-0.4, 0.2] if np.max(np.abs(summed_EAP_dict[0][use_elec_idx])) / 1000 < 1. else [-max_sig / 2, max_sig / 4]

        for jit_idx, jitter_STD in enumerate(jitter_STDs):

            sig = summed_EAP_dict[jitter_STD][use_elec_idx]

            filtered_signals = {}
            thresholds = {}
            lines = {}
            for sig_idx, signal_key in enumerate(signal_list):

                if signal_filter_params[signal_key] is None:
                    sig_ = sig
                else:
                    f0, f1 = signal_filter_params[signal_key]
                    sig_ = tools.filter_data(dt, sig, low_freq=f0, high_freq=f1)

                filtered_signals[signal_key] = sig_ - sig_[0]
                threshold = np.std(sig_) * 6
                # threshold = -np.min(sig_) / 3

                thresholds[signal_key] = threshold

                minima_idx = (np.diff(np.sign(np.diff(sig_))) > 0).nonzero()[0] + 1
                minima_idx = minima_idx[np.where(sig_[minima_idx] < -threshold)]

                if len(minima_idx) > 0:
                    latencies[signal_key][jit_idx, trial_num] = t[minima_idx[0]]
                else:
                    print("No peaks for {} jitter {} trial {}".format(signal_key, jitter_STD, trial_num))
                    latencies[signal_key][jit_idx, trial_num] = 0.

                if plot_trials:
                    ax_sig = fig.add_subplot(len(signal_list), num_cols,
                                             num_cols*sig_idx+jit_idx + 1,
                                             ylim=ylim, xlim=xlim)
                    if sig_idx + 1 == len(signal_list):
                        ax_sig.set_xlabel("Time [ms]")

                    if jit_idx == 0:
                        ax_sig.set_ylabel("mV")

                    if sig_idx == 0 :
                        ax_sig.set_title("Jitter STD: {} ms".format(jitter_STD), fontsize=12)
                    ax_sig.axhline(-threshold / 1000, ls=":", color=clr_dict[signal_key])

                    l, = ax_sig.plot(t, sig_ / 1000, lw=1, c=clr_dict[signal_key], clip_on=True)
                    lines[signal_key] = l
                    ax_sig.plot(t[minima_idx], sig_[minima_idx] / 1000, 'o', ms=2, c='r', clip_on=True)

                    if len(minima_idx) > 0 and signal_key is not "raw":
                        ax_sig.set_title("1st: {:1.02f} ms".format(t[minima_idx[0]]), fontsize=10)

                    ax_sig.axvline(mean_latency, ls="--", color="gray")

        if plot_trials:
            all_lines = [lines[signal_key] for signal_key in signal_list]
            all_line_names = [title_dict[signal_key] for signal_key in signal_list]
            fig.legend(all_lines, all_line_names,
                       loc="lower center", frameon=False, ncol=5)
            plt.savefig(join(root_folder, "jitter_%s_%d_trial_%d_dead_center.png" % (cell_name, num_cells, trial_num)))

    # Plot combined results
    plt.close("all")
    fig = plt.figure(figsize=[5, 7])
    fig.subplots_adjust(bottom=0.12, wspace=0.5, top=0.92, left=0.2, right=0.95)

    ax_mean = fig.add_subplot(2, 1, 1,
                             xlabel="Spike jitter STD (ms)",
                             ylim=[-5, +5], xlim=[0, np.max(jitter_STDs)])

    ax_std = fig.add_subplot(2, 1, 2,
                             xlabel="Spike jitter STD (ms)",
                             ylim=[0, 2.0], xlim=[0, np.max(jitter_STDs)])
    ax_mean.set_ylabel("Mean deviation from\ntrue center")
    ax_std.set_ylabel("STD of first peak")
    ax_mean.axhline(0, ls='--', c='gray')
    ax_std.plot([0, 2], [0, 2], c="gray", ls='--')
    lines = []
    line_names = []

    mean_dict = {"jitter_STDs": list(jitter_STDs),
                 "num_trial": num_trials,
                 "num_cells": num_cells}
    std_dict = {"jitter_STDs": list(jitter_STDs),
                "num_trial": num_trials,
                "num_cells": num_cells}

    for sig_idx, signal_key in enumerate(["spikes", "lfp", "lfp_hf"]):

        mean_deviations = []
        std_deviations = []
        for jit_idx, jitter_STD in enumerate(jitter_STDs):
            deviations = latencies[signal_key][jit_idx] - mean_latency
            ax_mean.scatter([jitter_STD] * num_trials, deviations,
                            c=clr_dict[signal_key], alpha=0.4, s=4)
            deviation_mean = np.average(deviations)
            deviation_std = np.std(deviations)

            mean_deviations.append(deviation_mean)
            std_deviations.append(deviation_std)
        mean_dict[signal_key] = mean_deviations
        std_dict[signal_key] = std_deviations
        l, = ax_mean.plot(jitter_STDs, mean_deviations, c=clr_dict[signal_key], ms=4, lw=2)
        ax_std.plot(jitter_STDs, std_deviations, c=clr_dict[signal_key], marker='x', ms=4, lw=2)
        lines.append(l)
        line_names.append(title_dict[signal_key])

    import json
    json.dump(mean_dict, file(join(root_folder, 'mean_latencies_{}_{}_dead_center.txt'.format(cell_name, num_cells)), 'w'))
    json.dump(std_dict, file(join(root_folder, 'std_latencies_{}_{}_dead_center.txt'.format(cell_name, num_cells)), 'w'))

    simplify_axes(fig.axes)
    fig.legend(lines, line_names, frameon=False, ncol=3, fontsize=11)
    fig.savefig(join(root_folder, "combined_data_{}_{}_dead_center.png".format(cell_name, num_cells)))
    fig.savefig(join(root_folder, "combined_data_{}_{}_dead_center.pdf".format(cell_name, num_cells)))


def plot_population(cell_name, num_cells):

    fig_name = cell_name
    electrode_parameters = return_electrode_parameters()
    electrode = LFPy.RecExtElectrode(**electrode_parameters)

    num_cells_to_plot = 5
    use_elec_idx = 10 if cell_name is "hbp_L4_SBC_bNAC219_1" else 7

    print use_elec_idx

    plt.close("all")

    fig = plt.figure(figsize=[9, 9])
    fig.subplots_adjust(wspace=0.5, right=0.98, left=0.1, hspace=0.5)

    ax_m = fig.add_axes([0.0, 0.0, 1.0, 1.], aspect=1, ylim=[-750, 10],
                        xlim=[-300,300], xticks=[], frameon=False,
                        rasterized=True, yticks=[])

    cell_idx_colors = lambda idx: plt.cm.viridis(1./(num_cells + 1) * idx)

    step = int(num_cells / num_cells_to_plot)

    for cell_number in range(num_cells)[::step]:
        plt.seed((random_seed + random_seed_shift[cell_name]) * cell_number)
        c = cell_idx_colors(np.random.randint(0, num_cells))

        cell = return_cell(cell_name=cell_name, cell_number=cell_number)
        zips = []
        from matplotlib.collections import PolyCollection
        for x, z in cell.get_pt3d_polygons():
            zips.append(list(zip(x, z)))
        polycol = PolyCollection(zips,
                                 edgecolors='none',
                                 facecolors=c)
        ax_m.add_collection(polycol)
    ax_m.plot([70, 120], [-500, -500], lw=3, c="k")
    ax_m.text(70, -520, "50 $\mu$m")

    ax_m.plot(electrode.x[use_elec_idx], electrode.z[use_elec_idx], 'D', c="gray", ms=14, clip_on=False)

    plt.savefig(join(root_folder, "cell_population_%s_%d.png" % (fig_name, num_cells)))
    plt.savefig(join(root_folder, "cell_population_%s_%d.pdf" % (fig_name, num_cells)))


def PopulationSerial(cell_name, num_cells):

    for cell_idx in range(0, num_cells):
        print("Simulating cell {} {} of {}".format(cell_name, cell_idx, num_cells))
        os.system("python cell_simulation.py %s %d" % (cell_name, cell_idx))

if __name__ == '__main__':
    num_cells = 500
    cell_name = ["hbp_L23_PC_cADpyr229_1", "hbp_L4_SS_cADpyr230_1", "hbp_L4_SBC_bNAC219_1"][1]
    # PopulationSerial(cell_name, num_cells)
    prope_EAP_synchronizations(cell_name, num_cells)
    # plot_population(cell_name, num_cells)