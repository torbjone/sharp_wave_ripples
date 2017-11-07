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
        plt.seed((random_seed + random_seed_shift[celltype]) * cell_number + 524)

        # print (random_seed + random_seed_shift[celltype]) * cell_number + 1
        c = cell_idx_colors(np.random.randint(0, tot_num_cells))
        # print c, cell_number, (random_seed + random_seed_shift[celltype]) * cell_number + 524

        # print np.random.random()
        cell = return_cell(celltype=celltype, conductance_type="passive", cell_number=cell_number)
        # print np.random.random()

        [ax_m.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                   c=c, zorder=0, rasterized=True)
         for idx in xrange(cell.totnsegs)]
        # [plt.plot(cell.xmid[idx], cell.zmid[idx], 'o', c=cell_idx_colors[idx], ms=12) for idx in cell_plot_idxs]
        # [ax.plot(cell.xmid[idx], cell.zmid[idx], 'o', c='orange', ms=2) for idx in cell.synidx]
        if cell_number == 0:
            cell, syn = make_input(cell, input_type)
            # print np.random.random()
            cell.simulate(rec_imem=True, rec_vmem=True)

            dz = electrode.z[1] - electrode.z[0]

            for syn_idx, syn in enumerate(cell.synapses):
                if cmap is not plt.cm.viridis:
                    color = cmap(0.5)
                else:
                    c = "r" if syn.kwargs["e"] > -60. else 'b'
                sptimes = cell.sptimeslist[syn_idx]
                ax_sp.plot(sptimes, np.ones(len(sptimes)) * cell.zmid[syn.idx], '.', color=c)

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
    dz = electrode.z[1] - electrode.z[0]

    cell = return_cell(celltype, conductance_type, 0)
    cell_name = 'swr_%s_%s_%s_%04d' % (input_type, celltype, conductance_type, 0)
    EAP = np.load(join(root_folder, celltype, "EAPs", "EAP_%s.npy" % cell_name))

    composed_signal_length = 150
    composed_tvec = np.arange(0, composed_signal_length / cell.dt + 1) * cell.dt

    composed_center = 40
    composed_center_idx = np.argmin(np.abs(composed_tvec - composed_center))

    jitter_STDs = np.array([0, 1, 2, 4, 8])

    summed_EAP_dict = {j: np.zeros((EAP.shape[0], len(composed_tvec)))
                       for j in jitter_STDs}


    for cell_idx in range(0, num_cells):
        fig_name = 'swr_%s_%s_%s_%04d' % (input_type, celltype, conductance_type, cell_idx)
        EAP = np.load(join(root_folder, celltype, "EAPs", "EAP_%s.npy" % fig_name))

        # tvec = np.arange(EAP.shape[1]) * cell.dt
        # plt.plot(tvec, EAP[6,:])
        # plt.savefig("testint.png")
        # plt.close("all")

        for jit_idx, jitter_STD in enumerate(jitter_STDs):
            jitter_idx = int(round(np.random.normal(0, jitter_STD / cell.dt)))
            t0 = composed_center_idx + jitter_idx - EAP.shape[1] / 2
            t1 = t0 + EAP.shape[1]
            summed_EAP_dict[jitter_STD][:, t0:t1] += EAP[:, :]

    num_cols = len(jitter_STDs)
    fig = plt.figure(figsize=[18, 8])
    xlim = [25, 55]

    for jit_idx, jitter_STD in enumerate(jitter_STDs):
        sig = summed_EAP_dict[jitter_STD]

        ax_lfp = plt.subplot(1, num_cols, jit_idx + 1,
                             title="Jitter STD: {} ms".format(jitter_STD),
                             frameon=False, ylabel='y [$\mu$m]',
                             xlabel='Time [ms]', xlim=xlim,
                             ylim=[np.min(electrode.z), 100])#, sharex=ax_e2)

        # img = ax_lfp.imshow(LFP, vmin=vmin, vmax=vmax, **img_dict)#, aspect=0.05)
        normalize = np.max(np.abs((sig[:, :] - sig[:, 0, None])))
        for idx in range(len(electrode.z)):
            y = electrode.z[idx] + (sig[idx] - sig[idx, 0]) / normalize * dz

            ax_lfp.plot(composed_tvec, y, lw=1, c='k', clip_on=True)

        ax_lfp.plot([xlim[1], xlim[1]], [-dz, 0], 'k', lw=4, clip_on=False)
        ax_lfp.text(xlim[1] + 2, -dz/2, "%1.2f mV" % (normalize / 1000.))
        ax_lfp.axvline(composed_center, ls="--", color="gray")

    fig_name = '%s_%s_%s' % (input_type, celltype, conductance_type)
    plt.savefig(join(root_folder, "jitter_swr_%s_%d.png" % (fig_name, num_cells)))




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
    num_cells = 5000
    input_types = ["waves"]#"changing_pathways"]#"tuft_pulse", "basal_pulse"][:]
    celltypes = ["hbp_L23_PC_cADpyr229_1", "hbp_L4_SS_cADpyr230_1"][-1:]

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