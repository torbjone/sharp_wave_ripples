import os
import sys
import numpy as np
from os.path import join
import matplotlib
matplotlib.use("AGG")
import pylab as plt
import tools
from plotting_convention import *
from main import return_cell, return_electrode_parameters, make_input, random_seed, random_seed_shift
import LFPy
from matplotlib import ticker

from mpi4py import MPI
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object



root_folder = ".."

def plot_num_cells_to_ax(num_cells, ax_m, ax_sp, ax_vsd,
                         celltype, electrode, input_type, tot_num_cells,
                         cmap=plt.cm.viridis):

    cell_idx_colors = lambda idx: cmap(1./(tot_num_cells + 1) * idx)

    step = int(tot_num_cells / num_cells)

    for cell_number in range(tot_num_cells)[::step]:
        # plt.seed((random_seed + random_seed_shift[celltype]) * cell_number + 524)
        plt.seed((random_seed + random_seed_shift[celltype]) * cell_number)

        # print (random_seed + random_seed_shift[celltype]) * cell_number + 1
        c = cell_idx_colors(np.random.randint(0, tot_num_cells))
        # print c, cell_number, (random_seed + random_seed_shift[celltype]) * cell_number + 524

        # print np.random.random()
        cell = return_cell(celltype=celltype, conductance_type="passive", cell_number=cell_number)
        # print np.random.random()

        [ax_m.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                   c=c, zorder=0, rasterized=True)
         for idx in xrange(cell.totnsegs)]
        [ax_m.plot(cell.xmid[idx], cell.zmid[idx], 'o', c=c, zorder=0, rasterized=True, ms=12   ) for idx in cell.somaidx]

        # [plt.plot(cell.xmid[idx], cell.zmid[idx], 'o', c=cell_idx_colors[idx], ms=12) for idx in cell_plot_idxs]
        # [ax.plot(cell.xmid[idx], cell.zmid[idx], 'o', c='orange', ms=2) for idx in cell.synidx]
        if cell_number == 0:
            cell, syn = make_input(cell, input_type, celltype)
            # print np.random.random()
            cell.simulate(rec_imem=True, rec_vmem=True)

            # dz = electrode.z[1] - electrode.z[0]

            if not ax_sp is None:

                for syn_idx, syn in enumerate(cell.synapses):
                    if cmap is not plt.cm.viridis:
                        color = cmap(0.5)
                    else:
                        c = "r" if syn.kwargs["e"] > -60. else 'b'
                    sptimes = cell.sptimeslist[syn_idx]
                    ax_sp.plot(sptimes, np.ones(len(sptimes)) * cell.zmid[syn.idx], '.', color=c)

    if not electrode is None:
        [ax_m.plot(electrode.x[idx], electrode.z[idx], 'D',  c="gray", ms=8, clip_on=False)
         for idx in xrange(len(electrode.x))]


def plot_simpler_LFP(conductance_type, cell_name, input_type, num_cells):

    electrode_parameters = return_electrode_parameters()
    electrode = LFPy.RecExtElectrode(**electrode_parameters)

    single_cell_fig_name = 'EAP_swr_%s_%s_%s_0000' % (input_type, cell_name, conductance_type)

    fig_name = 'summed_swr_%s_%s_%s' % (input_type, cell_name, conductance_type)

    single_cell_LFP = np.load(join(root_folder, cell_name, "EAPs", "%s.npy" % (single_cell_fig_name)))
    single_cell_LFP = single_cell_LFP - single_cell_LFP[:, 0, None]

    LFP = np.load(join(root_folder, cell_name, "EAPs", "%s_%d.npy" % (fig_name, num_cells)))
    LFP = LFP - LFP[:, 0, None]

    # print np.max(np.abs(LFP))
    # print np.max(np.abs(single_cell_LFP))

    timeres = 2**-4
    tvec = np.arange(LFP.shape[1]) * timeres

    plt.close('all')
    fig = plt.figure(figsize=(19, 10))

    plt.subplots_adjust(hspace=0.5, wspace=0.5, left=0.075, right=0.95, top=0.90)

    ax_sp = plt.subplot(141,  xlabel='Time [ms]', ylabel='y [$\mu$m]', title="Single-cell input spike-trains",
                       xlim=[0, tvec[-1]], ylim=[np.min(electrode.z), 100], frameon=False)

    ax_m = plt.subplot(142, aspect='equal', xlabel='x [$\mu$m]', ylabel='y [$\mu$m]',
                       title="Cell population (%d cells)\nand electrodes" % num_cells,
                        ylim=[np.min(electrode.z), 100], xticks=[], frameon=False, rasterized=True)

    num_cells_to_plot = np.min([100, num_cells])
    plot_num_cells_to_ax(num_cells_to_plot, ax_m, ax_sp, None, cell_name,
                         electrode, input_type, num_cells)

    mark_subplots([ax_sp], "A", xpos=-0.0, ypos=1.05)
    mark_subplots([ax_m], "B", xpos=-0.7, ypos=1.0)

    ax_lfp = plt.subplot(144, title="Population extracellular potentials", frameon=False, ylabel='y [$\mu$m]',
                         xlabel='Time [ms]', xlim=[0, tvec[-1]], ylim=[np.min(electrode.z), 100])#, sharex=ax_e2)
    dz = electrode.z[1] - electrode.z[0]

    # img = ax_lfp.imshow(LFP, vmin=vmin, vmax=vmax, **img_dict)#, aspect=0.05)
    normalize = np.max(np.abs((LFP[:, :] - LFP[:, 0, None])))
    for idx in range(len(electrode.z)):
        y = electrode.z[idx] + (LFP[idx] - LFP[idx, 0]) / normalize * dz

        ax_lfp.plot(tvec, y, lw=1, c='k', clip_on=False)

    ax_lfp.plot([tvec[-1], tvec[-1]], [-dz, 0], 'k', lw=4, clip_on=False)
    ax_lfp.text(tvec[-1] + 2, -dz/2, "%1.2f $\mu$V" % normalize)


    ax_lfp_single_cell = plt.subplot(143, title="Single-cell extracellular potentials",
                                     frameon=False, ylabel='y [$\mu$m]',
                                     xlabel='Time [ms]', xlim=[0, tvec[-1]], ylim=[np.min(electrode.z), 100])
    # vmax = 0.01#np.max(np.abs(single_cell_LFP)) #/ 5
    # vmin = -vmax

    # img = ax_lfp_single_cell.imshow(single_cell_LFP, vmin=vmin, vmax=vmax, **img_dict)#, aspect=0.05)
    normalize = np.max(np.abs((single_cell_LFP[:, :] - single_cell_LFP[:, 0, None])))
    dz = electrode.z[1] - electrode.z[0]
    for idx in range(len(electrode.z)):
        y = electrode.z[idx] + (single_cell_LFP[idx] - single_cell_LFP[idx, 0]) / normalize * dz
        ax_lfp_single_cell.plot(tvec, y, lw=1, c='k', clip_on=False)

    ax_lfp_single_cell.plot([tvec[-1], tvec[-1]], [-dz, 0], 'k', lw=4, clip_on=False)
    ax_lfp_single_cell.text(tvec[-1] + 2, -dz/2, "%1.2f $\mu$V" % normalize)

    simplify_axes(fig.axes)

    mark_subplots([ax_lfp_single_cell, ax_lfp], "CD", xpos=-0.1, ypos=1.1)
    plt.savefig(join(root_folder, '%s_%d.png' % (fig_name, num_cells)))

def sum_EAPs(conductance_type, celltype, input_type, num_cells):

    summed_LFP = np.zeros((0,0))
    for cell_idx in range(0, num_cells):
        fig_name = 'swr_%s_%s_%s_%04d' % (input_type, celltype, conductance_type, cell_idx)
        LFP = np.load(join(root_folder, celltype, "EAPs", "EAP_%s.npy" % fig_name))
        if summed_LFP.shape != LFP.shape:
            summed_LFP = LFP
        else:
            summed_LFP += LFP
    fig_name = '%s_%s_%s' % (input_type, celltype, conductance_type)
    np.save(join(root_folder, celltype, "EAPs", "summed_swr_%s_%d.npy" % (fig_name, num_cells)), summed_LFP)


def prope_EAP_synchronizations(conductance_type, celltype, input_type, num_cells):


    electrode_parameters = return_electrode_parameters()
    electrode = LFPy.RecExtElectrode(**electrode_parameters)

    cell = return_cell(celltype, conductance_type, 0)
    cell_name = 'swr_%s_%s_%s_%04d' % (input_type, celltype, conductance_type, 0)
    EAP = np.load(join(root_folder, celltype, "EAPs", "EAP_%s.npy" % cell_name))

    composed_signal_length = 300
    composed_tvec = np.arange(0, composed_signal_length / cell.dt + 1) * cell.dt

    composed_center = 100
    composed_center_idx = np.argmin(np.abs(composed_tvec - composed_center))

    t = composed_tvec - composed_center

    jitter_STDs = np.array([0, 0.5, 1, 2])

    use_elec_idx = 9 # 7

    np.random.seed(12345)

    summed_EAP_dict = {j: np.zeros((EAP.shape[0], len(composed_tvec)))
                       for j in jitter_STDs}

    for cell_idx in range(0, num_cells):
        cell_name = 'swr_%s_%s_%s_%04d' % (input_type, celltype, conductance_type, cell_idx)
        # print cell_name
        EAP = np.load(join(root_folder, celltype, "EAPs", "EAP_%s.npy" % cell_name))

        # tvec = np.arange(EAP.shape[1]) * cell.dt
        # plt.plot(tvec, EAP[use_elec_idx,:])
        # plt.savefig("testint_{}.png".format(cell_name))
        # plt.close("all")

        for jit_idx, jitter_STD in enumerate(jitter_STDs):
            jitter_idx = int(round(np.random.normal(0, jitter_STD / cell.dt)))
            t0 = composed_center_idx + jitter_idx - EAP.shape[1] / 2
            t1 = t0 + EAP.shape[1]
            summed_EAP_dict[jitter_STD][:, t0:t1] += EAP[:, :]

    num_cols = len(jitter_STDs)
    fig = plt.figure(figsize=[18, 9])
    fig.subplots_adjust(wspace=0.5, right=0.98, left=0.2)

    ax_m = fig.add_axes([0, 0.05, 0.2, 0.9], aspect=1, xlabel='x [$\mu$m]', ylabel='y [$\mu$m]',
                       title="Cell population (%d cells)\nand electrodes" % num_cells,
                        ylim=[np.min(electrode.z), 100], xticks=[], frameon=False, rasterized=True)

    num_cells_to_plot = np.min([5, num_cells])
    plot_num_cells_to_ax(num_cells_to_plot, ax_m, None, None, celltype,
                         None, input_type, num_cells)

    ax_m.plot(electrode.x[use_elec_idx], electrode.z[use_elec_idx], 'D', c="gray", ms=8, clip_on=False)

    xlim = [-5, 10]
    ylim = [-0.8, 0.4]

    for jit_idx, jitter_STD in enumerate(jitter_STDs):
        sig = summed_EAP_dict[jitter_STD]
        filtered_wband = tools.filter_data(cell.dt, sig, low_freq=20.0, high_freq=3000.)
        filtered_spikes = tools.filter_data(cell.dt, sig, low_freq=600.0, high_freq=3000.)
        filtered_lfp = tools.filter_data(cell.dt, sig, low_freq=20.0, high_freq=600.)


        ax_raw = plt.subplot(4, num_cols, 0 + jit_idx + 1,
                             title="Jitter STD: {} ms".format(jitter_STD),
                             ylabel='mV', ylim=ylim,
                             # xlabel='Time [ms]',
                             xlim=xlim)

        ax_wband = plt.subplot(4, num_cols, 4 + jit_idx + 1,
                             # title="Jitter STD: {} ms".format(jitter_STD),
                             ylabel='mV', ylim=ylim,
                             # xlabel='Time [ms]',
                             xlim=xlim)

        ax_spikes = plt.subplot(4, num_cols, 8 + jit_idx + 1,
                             # title="Jitter STD: {} ms".format(jitter_STD),
                             ylabel='mV', ylim=ylim,
                             # xlabel='Time [ms]',
                             xlim=xlim)

        ax_lfp = plt.subplot(4, num_cols, 12 + jit_idx + 1,
                             # title="Jitter STD: {} ms".format(jitter_STD),
                             ylabel='mV', ylim=ylim,
                             xlabel='Time [ms]', xlim=xlim)
#                             ylim=[np.min(electrode.z), 100])#, sharex=ax_e2)

        # img = ax_lfp.imshow(LFP, vmin=vmin, vmax=vmax, **img_dict)#, aspect=0.05)
        # normalize = np.max(np.abs((sig[:, :] - sig[:, 0, None])))
        # for idx in range(len(electrode.z)):
        #     if idx != 7:
        #         continue
        # y = electrode.z[idx] + (sig[idx] - sig[idx, 0]) / normalize * dz
        y_raw = sig[use_elec_idx] - sig[use_elec_idx, 0]
        y_wband = filtered_wband[use_elec_idx] - filtered_wband[use_elec_idx, 0]
        y_spikes = filtered_spikes[use_elec_idx] - filtered_spikes[use_elec_idx, 0]
        y_lfp = filtered_lfp[use_elec_idx] - filtered_lfp[use_elec_idx, 0]

        threshold_lfp = np.max(np.abs(y_lfp)) / 3
        threshold_raw = np.max(np.abs(y_raw)) / 3
        threshold_wband = np.max(np.abs(y_wband)) / 3
        threshold_spikes = np.max(np.abs(y_spikes)) / 3

        minima_idx_raw = (np.diff(np.sign(np.diff(y_raw))) > 0).nonzero()[0] + 1
        minima_idx_raw = minima_idx_raw[np.where(y_raw[minima_idx_raw] < -threshold_raw)]

        minima_idx_wband = (np.diff(np.sign(np.diff(y_wband))) > 0).nonzero()[0] + 1
        minima_idx_wband = minima_idx_wband[np.where(y_wband[minima_idx_wband] < -threshold_wband)]

        minima_idx_spikes = (np.diff(np.sign(np.diff(y_spikes))) > 0).nonzero()[0] + 1
        minima_idx_spikes = minima_idx_spikes[np.where(y_spikes[minima_idx_spikes] < -threshold_spikes)]

        minima_idx_lfp = (np.diff(np.sign(np.diff(y_lfp))) > 0).nonzero()[0] + 1
        minima_idx_lfp = minima_idx_lfp[np.where(y_lfp[minima_idx_lfp] < -threshold_lfp)]


        ax_raw.axhline(-threshold_raw / 1000, ls="--", color="pink")
        ax_wband.axhline(-threshold_wband / 1000, ls="--", color="pink")
        ax_spikes.axhline(-threshold_spikes / 1000, ls="--", color="pink")
        ax_lfp.axhline(-threshold_lfp / 1000, ls="--", color="pink")

        l1, = ax_raw.plot(t, y_raw / 1000, lw=1, c='gray', clip_on=True)
        l2, = ax_wband.plot(t, y_wband / 1000, lw=1, c='k', clip_on=True)
        l3, = ax_spikes.plot(t, y_spikes / 1000, lw=1, c='g', clip_on=True)
        l4, = ax_lfp.plot(t, y_lfp / 1000, lw=1, c='b', clip_on=True)

        ax_raw.plot(t[minima_idx_raw], y_raw[minima_idx_raw] / 1000, 'o', ms=2, c='r', clip_on=True)
        ax_wband.plot(t[minima_idx_wband], y_wband[minima_idx_wband] / 1000, 'o', ms=2, c='r', clip_on=True)
        ax_spikes.plot(t[minima_idx_spikes], y_spikes[minima_idx_spikes] / 1000, 'o', ms=2, c='r', clip_on=True)
        ax_lfp.plot(t[minima_idx_lfp], y_lfp[minima_idx_lfp] / 1000, 'o', ms=2, c='r', clip_on=True)


        # ax_lfp.plot(composed_tvec - composed_center, y_filt, lw=1, c='r', clip_on=True)

        # ax_lfp.plot([xlim[1], xlim[1]], [-normalize, 0], 'k', lw=4, clip_on=False)
        # ax_lfp.text(xlim[1] + 2, -dz/2, "%1.2f mV" % (normalize / 1000.))
        ax_raw.axvline(0, ls="--", color="gray")
        ax_wband.axvline(0, ls="--", color="gray")
        ax_spikes.axvline(0, ls="--", color="gray")
        ax_lfp.axvline(0, ls="--", color="gray")

    fig.legend([l1, l2, l3, l4], ["Raw", "Wideband (20-3000 Hz)",
                                  "Spikes (600 - 3000 Hz)",
                                  "LFP (20-600 Hz)"],
               loc="lower center", frameon=False, ncol=4)
    fig_name = '%s_%s_%s' % (input_type, celltype, conductance_type)
    plt.savefig(join(root_folder, "jitter_swr_%s_%d_peaks.png" % (fig_name, num_cells)))




def Population(input_types, celltypes, num_cells):
    """ Run with
        mpirun -np 4 python example_mpi.py
    """

    class Tags:
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries

    num_workers = size - 1

    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        # num_cells = 400

        # celltypes = ['L23', "L5"]
        # input_types = ["changing_pathways"]#"tuft_pulse", "basal_pulse"]

        num_tasks = len(celltypes) * len(input_types) * num_cells

        for cell_name in celltypes:
            for input_type in input_types:
                for cell_idx in range(0, num_cells):
                    task += 1
                    sent = False
                    while not sent:
                        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                        source = status.Get_source()
                        tag = status.Get_tag()
                        if tag == tags.READY:
                            comm.send([cell_name, input_type, cell_idx], dest=source, tag=tags.START)
                            print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                            sent = True
                        elif tag == tags.DONE:
                            print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                        elif tag == tags.ERROR:
                            print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                            for worker in range(1, num_workers + 1):
                                comm.send([None, None, None], dest=worker, tag=tags.EXIT)
                            sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None, None, None], dest=worker, tag=tags.EXIT)

        print("\033[95m Master finishing\033[0m")
    else:

        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [cell_name, input_type, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s cell %d\033[0m" % (rank, cell_name, input_type, cell_idx)
                # try:
                print "python main.py %s %s %d" % (cell_name, input_type, cell_idx)
                os.system("python main.py %s %s %d" % (cell_name, input_type, cell_idx))
                # except:
                #     print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                #     comm.send(None, dest=0, tag=tags.ERROR)
                #     sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)

def PopulationSerial(input_types, celltypes, num_cells):
    """ Run with
        mpirun -np 4 python example_mpi.py
    """

    for cell_name in celltypes:
        for input_type in input_types:
            for cell_idx in range(0, num_cells):
                print cell_idx, num_cells
                os.system("python main.py %s %s %d" % (cell_name, input_type, cell_idx))


def sum_all(input_types, celltypes, num_cells):

    conductance_type = "active"
    for celltype in celltypes:
        for input_type in input_types:
            sum_EAPs(conductance_type, celltype, input_type, num_cells)

def plot_all(input_types, celltypes, num_cells):

    conductance_type = "active"
    for celltype in celltypes:
        for input_type in input_types:
            plot_simpler_LFP(conductance_type, celltype, input_type, num_cells)


if __name__ == '__main__':
    num_cells = 500
    input_types = ["waves"]#"changing_pathways"]#"tuft_pulse", "basal_pulse"][:]
    celltypes = ["hbp_L23_PC_cADpyr229_1", "hbp_L4_SS_cADpyr230_1", "hbp_L4_SBC_bNAC219_1"][-1:]

    # if len(sys.argv) == 2 and sys.argv[1] == "MPI":
    #     print "Running population with %d cells" % num_cells
    #     TO RUN SIMULATION, write in terminal "mpirun -np 4 python mpi_run.py"

    # PopulationSerial(input_types, celltypes, num_cells)

    # else:
    #     print "Summing and plotting results"
    #     TO SUM SIGNALS AND PLOT RESULTS
    #     if rank == 0:
    # sum_all(input_types, celltypes, num_cells)
    # plot_all(input_types, celltypes, num_cells)
    prope_EAP_synchronizations("active", celltypes[0], input_types[0], num_cells)