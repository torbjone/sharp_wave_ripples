import os
import sys
import numpy as np
from os.path import join
import matplotlib
matplotlib.use("AGG")
import pylab as plt
import tools
from plotting_convention import *
from cell_simulation import return_cell, return_electrode_parameters, make_input, random_seed, random_seed_shift
import LFPy
from matplotlib import ticker


root_folder = ".."


def prope_EAP_synchronizations(cell_name, input_type, num_cells):

    fig_name = '%s_%s_%s' % (input_type, cell_name)
    num_trials = 100

    cell = return_cell(cell_name, 0)
    dt = cell.dt
    cell_name = 'swr_%s_%04d' % (cell_name, 0)
    EAP = np.load(join(root_folder, cell_name, "EAPs", "EAP_%s.npy" % cell_name))

    composed_signal_length = 300
    composed_tvec = np.arange(0, composed_signal_length / dt + 1) * dt

    composed_center = 100
    composed_center_idx = np.argmin(np.abs(composed_tvec - composed_center))

    mean_latency = 10

    t = composed_tvec - composed_center + mean_latency

    jitter_STDs = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    num_cols = len(jitter_STDs)
    # use_elec_idx = 10 if celltype is "hbp_L4_SBC_bNAC219_1" else 6
    use_elec_idx = 18 if cell_name is "hbp_L4_SBC_bNAC219_1" else 6
    print use_elec_idx

    latencies = {
        "raw": np.zeros((len(jitter_STDs), num_trials)),
        "wband": np.zeros((len(jitter_STDs), num_trials)),
        "spikes": np.zeros((len(jitter_STDs), num_trials)),
        "mua": np.zeros((len(jitter_STDs), num_trials)),
        "lfp": np.zeros((len(jitter_STDs), num_trials)),
    }

    clr_dict = {"raw": "gray",
                "wband": "k",
                "spikes": "g",
                "mua": "c",
                "lfp": "b",
                }

    plot_trials = False

    for trial_num in range(num_trials):
        print trial_num

        np.random.seed(1234 + trial_num)

        summed_EAP_dict = {j: np.zeros((EAP.shape[0], len(composed_tvec)))
                           for j in jitter_STDs}

        for cell_idx in range(0, num_cells):

            cell_name = 'swr_%s_%04d' % (cell_name, cell_idx)
            EAP = np.load(join(root_folder, cell_name, "EAPs", "EAP_%s.npy" % cell_name))

            for jit_idx, jitter_STD in enumerate(jitter_STDs):
                jitter_idx = int(round(np.random.normal(0, jitter_STD / cell.dt)))
                t0 = composed_center_idx + jitter_idx - EAP.shape[1] / 2
                t1 = t0 + EAP.shape[1]
                summed_EAP_dict[jitter_STD][:, t0:t1] += EAP[:, :]

        for jit_idx, jitter_STD in enumerate(jitter_STDs):
            sig = summed_EAP_dict[jitter_STD]
            filtered_wband = tools.filter_data(cell.dt, sig, low_freq=20.0, high_freq=3000.)
            filtered_spikes = tools.filter_data(cell.dt, sig, low_freq=600.0, high_freq=3000.)
            # filtered_spikes = tools.filter_data_butterworth(cell.dt, sig, low_freq=600.0, high_freq=3000.)
            # filtered_MUA = -np.abs(tools.filter_data_butterworth(cell.dt, sig, low_freq=600.0, high_freq=3000.))
            # filtered_MUA = tools.filter_data(cell.dt, filtered_MUA, low_freq=20.0, high_freq=600.)
            filtered_lfp = tools.filter_data(cell.dt, sig, low_freq=20.0, high_freq=600.)

            y_raw = sig[use_elec_idx] - sig[use_elec_idx, 0]
            y_wband = filtered_wband[use_elec_idx] - filtered_wband[use_elec_idx, 0]
            y_spikes = filtered_spikes[use_elec_idx] - filtered_spikes[use_elec_idx, 0]
            # y_MUA = filtered_MUA[use_elec_idx] - filtered_MUA[use_elec_idx, 0]
            y_lfp = filtered_lfp[use_elec_idx] - filtered_lfp[use_elec_idx, 0]

            threshold_raw = np.max(np.abs(y_raw)) / 3
            threshold_wband = np.max(np.abs(y_wband)) / 3
            threshold_spikes = np.max(np.abs(y_spikes)) / 3
            # threshold_MUA = np.max(np.abs(y_MUA)) / 3
            threshold_lfp = np.max(np.abs(y_lfp)) / 3

            minima_idx_raw = (np.diff(np.sign(np.diff(y_raw))) > 0).nonzero()[0] + 1
            minima_idx_raw = minima_idx_raw[np.where(y_raw[minima_idx_raw] < -threshold_raw)]

            minima_idx_wband = (np.diff(np.sign(np.diff(y_wband))) > 0).nonzero()[0] + 1
            minima_idx_wband = minima_idx_wband[np.where(y_wband[minima_idx_wband] < -threshold_wband)]

            minima_idx_spikes = (np.diff(np.sign(np.diff(y_spikes))) > 0).nonzero()[0] + 1
            minima_idx_spikes = minima_idx_spikes[np.where(y_spikes[minima_idx_spikes] < -threshold_spikes)]

            # minima_idx_MUA = (np.diff(np.sign(np.diff(y_MUA))) > 0).nonzero()[0] + 1
            # minima_idx_MUA = minima_idx_MUA[np.where(y_MUA[minima_idx_MUA] < -threshold_MUA)]
            # minima_idx_MUA = np.array([np.argmax(np.abs(y_MUA))])

            minima_idx_lfp = (np.diff(np.sign(np.diff(y_lfp))) > 0).nonzero()[0] + 1
            minima_idx_lfp = minima_idx_lfp[np.where(y_lfp[minima_idx_lfp] < -threshold_lfp)]

            # for min_list in [minima_idx_raw, minima_idx_wband,
            #                  minima_idx_spikes, minima_idx_MUA, minima_idx_lfp]:

            try:
                latencies["raw"][jit_idx, trial_num] = t[minima_idx_raw[0]]
                latencies["wband"][jit_idx, trial_num] = t[minima_idx_wband[0]]
                latencies["spikes"][jit_idx, trial_num] = t[minima_idx_spikes[0]]
                # latencies["mua"][jit_idx, trial_num] = t[minima_idx_MUA[0]]
                latencies["lfp"][jit_idx, trial_num] = t[minima_idx_lfp[0]]
            except:
                latencies["raw"][jit_idx, trial_num] = 0.
                latencies["wband"][jit_idx, trial_num] = 0.
                latencies["spikes"][jit_idx, trial_num] = 0.
                # latencies["mua"][jit_idx, trial_num] = 0.
                latencies["lfp"][jit_idx, trial_num] = 0.

            if plot_trials:
                plt.close("all")
                fig = plt.figure(figsize=[18, 9])
                fig.subplots_adjust(wspace=0.5, right=0.98, left=0.1, hspace=0.5)
                ax_raw = plt.subplot(4, num_cols, 0 + jit_idx + 1,
                                     ylim=ylim,
                                     xlim=xlim)
                ax_wband = plt.subplot(4, num_cols, (num_cols) * 1 + jit_idx + 1,
                                     ylim=ylim,
                                     xlim=xlim)

                ax_spikes = plt.subplot(4, num_cols, (num_cols ) * 2 + jit_idx + 1,
                                     ylim=ylim,
                                     xlim=xlim)

                ax_lfp = plt.subplot(4, num_cols, (num_cols) * 3 + jit_idx + 1,
                                     ylim=ylim,
                                     xlabel='Time [ms]', xlim=xlim)

                max_sig = np.max(np.abs(summed_EAP_dict[0][use_elec_idx])) / 1000
                xlim = [5, 15]
                ylim = [-0.4, 0.2]if np.max(np.abs(summed_EAP_dict[0][use_elec_idx])) / 1000 < 1. else [-max_sig / 2, max_sig / 4]

                if jit_idx == 0:
                    ax_raw.set_ylabel("RAW\nmV")
                    ax_wband.set_ylabel("Wideband\nmV")
                    ax_spikes.set_ylabel("Spikes\nmV")
                    ax_lfp.set_ylabel("LFP\nmV")
                ax_raw.set_title("Jitter STD: {} ms".format(jitter_STD), fontsize=12)

                ax_raw.axhline(-threshold_raw / 1000, ls=":", color="pink")
                ax_wband.axhline(-threshold_wband / 1000, ls=":", color="pink")
                ax_spikes.axhline(-threshold_spikes / 1000, ls=":", color="pink")
                ax_lfp.axhline(-threshold_lfp / 1000, ls=":", color="pink")

                l1, = ax_raw.plot(t, y_raw / 1000, lw=1, c='gray', clip_on=True)
                l2, = ax_wband.plot(t, y_wband / 1000, lw=1, c='k', clip_on=True)
                l3, = ax_spikes.plot(t, y_spikes / 1000, lw=1, c='g', clip_on=True)
                # l3b, = ax_spikes.plot(t, y_MUA / 1000, lw=1, c='cyan', clip_on=True)
                l4, = ax_lfp.plot(t, y_lfp / 1000, lw=1, c='b', clip_on=True)

                ax_raw.plot(t[minima_idx_raw], y_raw[minima_idx_raw] / 1000, 'o', ms=2, c='r', clip_on=True)
                ax_wband.plot(t[minima_idx_wband], y_wband[minima_idx_wband] / 1000, 'o', ms=2, c='r', clip_on=True)
                ax_spikes.plot(t[minima_idx_spikes], y_spikes[minima_idx_spikes] / 1000, 'o', ms=2, c='r', clip_on=True)
                # ax_spikes.plot(t[minima_idx_MUA], y_MUA[minima_idx_MUA] / 1000, 'o', ms=2, c='orange', clip_on=True)
                ax_lfp.plot(t[minima_idx_lfp], y_lfp[minima_idx_lfp] / 1000, 'o', ms=2, c='r', clip_on=True)

                ax_raw.axvline(mean_latency, ls="--", color="gray")
                ax_wband.axvline(mean_latency, ls="--", color="gray")
                ax_spikes.axvline(mean_latency, ls="--", color="gray")
                ax_lfp.axvline(mean_latency, ls="--", color="gray")
        if plot_trials:
            fig.legend([l1, l2, l3, l4], ["Raw", "Wideband (20-3000 Hz)",
                                          "Spikes (600 - 3000 Hz)",
                                          # "MUA low-pass(|600 - 5000 Hz|)",
                                          "LFP (20-600 Hz)"],
                       loc="lower center", frameon=False, ncol=5)
            plt.savefig(join(root_folder, "jitter_swr_%s_%d_trial_%d.png" % (fig_name, num_cells, trial_num)))

    # Plot combined results
    plt.close("all")
    fig = plt.figure(figsize=[5, 7])
    fig.subplots_adjust(bottom=0.12, wspace=0.5, top=0.92, left=0.2, right=0.95)
    title_dict = {"raw": "Raw data\nNo filter",
                  "wband": "Wide band\n20-3000 Hz",
                  "spikes": "Spike\n600-3000 Hz",
                  #"mua": "MUA\n20-600 Hz of |600-3000 Hz|",
                  "lfp": "LFP\n20-600 Hz",
                  }
    ax_mean = fig.add_subplot(2, 1, 1,
                             xlabel="Spike jitter STD (ms)",
                             ylim=[-5, +5])

    ax_std = fig.add_subplot(2, 1, 2,
                             xlabel="Spike jitter STD (ms)",
                             ylim=[0, +1.5])
    ax_mean.set_ylabel("Mean deviation from\ntrue center")
    ax_std.set_ylabel("STD of first peak")
    ax_mean.axhline(0, ls='--', c='gray')

    lines = []
    line_names = []
    for sig_idx, signal_key in enumerate(["spikes", "lfp"]):


        mean_deviations = []
        std_deviations = []
        for jit_idx, jitter_STD in enumerate(jitter_STDs):
            ax_mean.scatter([jitter_STD] * num_trials, (latencies[signal_key][jit_idx] - mean_latency), c=clr_dict[signal_key], alpha=0.4, s=4)
            deviation_mean = np.average(latencies[signal_key][jit_idx] - mean_latency)
            deviation_std = np.std(latencies[signal_key][jit_idx] - mean_latency)

            mean_deviations.append(deviation_mean)
            std_deviations.append(deviation_std)

        l = ax_mean.errorbar(jitter_STDs, mean_deviations, yerr=std_deviations, c=clr_dict[signal_key], ms=4, lw=2)
        ax_std.plot(jitter_STDs, std_deviations, c=clr_dict[signal_key], marker='x', ms=4, lw=2)
        lines.append(l)
        line_names.append(title_dict[signal_key])

    simplify_axes(fig.axes)
    fig.legend(lines, line_names, frameon=False, ncol=3)
    fig.savefig(join(root_folder, "combined_data_{}_{}.png".format(fig_name, num_cells)))
    fig.savefig(join(root_folder, "combined_data_{}_{}.pdf".format(fig_name, num_cells)))


def plot_population(cell_name, num_cells):

    fig_name = cell_name
    electrode_parameters = return_electrode_parameters()
    electrode = LFPy.RecExtElectrode(**electrode_parameters)

    num_cells_to_plot = 100

    use_elec_idx = 10 if cell_name is "hbp_L4_SBC_bNAC219_1" else 6
    # use_elec_idx = 9 if celltype is "hbp_L4_SBC_bNAC219_1" else 6
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
        print("Simulating cell %d of %d".format(cell_idx, num_cells))
        os.system("python cell_simulation.py %s %d" % (cell_name, cell_idx))

if __name__ == '__main__':
    num_cells = 100
    cell_name = ["hbp_L23_PC_cADpyr229_1", "hbp_L4_SS_cADpyr230_1", "hbp_L4_SBC_bNAC219_1"][1]
    PopulationSerial(cell_name, num_cells)
    prope_EAP_synchronizations(cell_name, num_cells)
    plot_population(cell_name, num_cells)