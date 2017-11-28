#!/usr/bin/env python

import sys
import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import pylab as plt
import neuron
import LFPy
from neuron import h


root_folder = '..'
layer_thickness = 100
random_seed = 123
random_seed_shift = {
                     "hbp_L4_SS_cADpyr230_1": 5,
                     'hbp_L5_TTPC2_cADpyr232_1': 6,
                     "hbp_L4_SBC_bNAC219_1": 7,}

cell_type_weight_scale = {"hbp_L4_SS_cADpyr230_1": 0.1,
                          'hbp_L5_TTPC2_cADpyr232_1': 1.0,
                          "hbp_L4_SBC_bNAC219_1": 0.007,}


def hbp_return_cell(cell_folder, end_T, dt, start_T, v_init=-65.):

    cell_name = os.path.split(cell_folder)[1]
    cwd = os.getcwd()
    os.chdir(cell_folder)
    # print "Simulating ", cell_name

    f = file("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    f = file("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    f = file("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    f = file(os.path.join("synapses", "synapses.hoc"), 'r')
    synapses = get_templatename(f)
    f.close()
    print('Loading constants')
    neuron.h.load_file('constants.hoc')
    # with suppress_stdout_stderr():

    if not hasattr(neuron.h, morphology):
        neuron.h.load_file(1, "morphology.hoc")

    if not hasattr(neuron.h, biophysics):
        neuron.h.load_file(1, "biophysics.hoc")
    #get synapses template name
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, os.path.join('synapses', 'synapses.hoc'))

    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    morphologyfile = os.listdir('morphology')[0]#glob('morphology\\*')[0]

    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=join('morphology', morphologyfile),
                     templatefile=os.path.join('template.hoc'),
                     templatename=templatename,
                     passive=False,
                     templateargs=0,
                     tstop=end_T,
                     tstart=start_T,
                     dt=dt,
                     # dt=dt,
                     extracellular=False,
                     celsius=37,
                     v_init=v_init,
                     pt3d=True,
                     )
    os.chdir(cwd)
    cell.set_rotation(z=np.pi/2, x=np.pi/2)
    # print cell.tstopms, cell.dt
    return cell


def initialize_population(num_cells, celltype):
    print "Initializing cell positions and rotations ..."
    cell_density = 200000. * 1e-9 #  cells / um^3

    pop_radius = np.sqrt(num_cells / (cell_density * np.pi * layer_thickness))
    print num_cells, pop_radius

    x_y_z_rot = np.zeros((num_cells, 4))

    for cell_number in range(num_cells):
        plt.seed((random_seed + random_seed_shift[celltype]) * cell_number + 1)
        rotation = 2*np.pi*np.random.random()

        z = layer_thickness * (np.random.random() - 0.5)
        x, y = 2 * (np.random.random(2) - 0.5) * pop_radius
        while np.sqrt(x**2 + y**2) > pop_radius:
            x, y = 2 * (np.random.random(2) - 0.5) * pop_radius
        x_y_z_rot[cell_number, :] = [x, y, z, rotation]

    r = np.array([np.sqrt(d[0]**2 + d[1]**2) for d in x_y_z_rot])

    argsort = np.argsort(r)
    x_y_z_rot = x_y_z_rot[argsort]

    np.save(join(root_folder, 'x_y_z_rot_%d_%s.npy' % (num_cells, celltype)), x_y_z_rot)
    plt.close("all")
    plt.subplot(121, xlabel='x', ylabel='y', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 1], edgecolor='none', s=2)

    plt.subplot(122, xlabel='x', ylabel='z', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 2], edgecolor='none', s=2)

    plt.savefig(join(root_folder, 'population_%d_%s.png' % (num_cells, celltype)))


def make_synapse(cell, weight, input_idx, input_spike_train, e=0.):

    synapse_parameters = {
        'idx': input_idx,
        'e': e,
        'syntype': 'Exp2Syn',
        # 'tau': 2.,
        'tau1' : 1.,                #Time constant, rise
        'tau2' : 3.,                #Time constant, decay
        # 'tau2': 2,
        'weight': weight,
        'record_current': False,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(input_spike_train)
    return cell, synapse

def return_cell(cell_name, cell_number=None):
    ### MAKING THE CELL

    model_folder = join("/home", "tone", "work", "hbp_cell_models")
    # from hbp_cell_models import return_cell as hbp_return_cell


    v_rest = -75.
    cell_name = cell_name.replace("hbp_", "")
    cell_folder = join(model_folder, "models", cell_name)
    end_T = 150
    dt = 2**-5
    start_T = -200
    cell = hbp_return_cell(cell_folder, end_T, dt, start_T, v_init=v_rest)
    cell.set_rotation(x=0)

    if cell_number is not None:
        cell_x_y_z_rotation = np.load(join(root_folder, 'x_y_z_rot_%d_%s.npy' % (10000, cell_name)))
        cell.set_rotation(z=cell_x_y_z_rotation[cell_number][3])

        z_shift = np.max(cell.zend) + layer_thickness / 2
        cell.set_pos(x=cell_x_y_z_rotation[cell_number][0],
                     y=cell_x_y_z_rotation[cell_number][1],
                     z=cell_x_y_z_rotation[cell_number][2] - z_shift)
        if np.max(cell.zend) > 0:
            raise RuntimeError("Cell reaches above cortex")

    return cell


def make_input(cell, cell_name):

    weight = 0.002 * cell_type_weight_scale[cell_name]

    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=100)
                                             # z_min=cell.zmid[0]-100,
                                             # z_max=cell.zmid[0]+100)
    # Input is centered around:
    input_center = 30
    input_std = 0.25
    for input_idx in input_idxs:
        input_spike_train = np.random.normal(input_center, input_std,
                                             size=1)
        cell, synapse = make_synapse(cell,
                                     weight * np.random.normal(1., 0.2),
                                     input_idx, input_spike_train)

    return cell, synapse


def return_electrode_parameters():
    # Making x,y,z coordinates of three electrodes
    elec_z = np.linspace(-600, 0, 13)#np.linspace(-900, 0, 9)
    elec_x = np.ones(len(elec_z)) * (0)
    elec_y = np.zeros(len(elec_z))
    electrode_parameters = {
        'sigma': 0.3,       # extracellular conductivity
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
        'method': 'soma_as_point',
    }
    return electrode_parameters


def single_cell_compare(cell_number, cell_name, plot=False):

    fig_name = 'swr_%s_%04d' % (cell_name, cell_number)
    plt.seed((random_seed + random_seed_shift[cell_name]) * cell_number)
    cell = return_cell(cell_name, cell_number)
    cell, syn_input = make_input(cell, cell_name)
    cell.simulate(rec_imem=True, rec_vmem=True)
    electrode_parameters = return_electrode_parameters()

    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    LFP = 1000 * electrode.LFP

    if not os.path.isdir(join(root_folder, cell_name, "EAPs")):
        os.mkdir(join(root_folder, cell_name))
        os.mkdir(join(root_folder, cell_name, "EAPs"))

    spike_time_idx = np.argmax(cell.somav)
    tlim = [cell.tvec[spike_time_idx] - 15, cell.tvec[spike_time_idx] + 15]
    spike_window_idx = [np.argmin(np.abs(cell.tvec - lim)) for lim in tlim]

    np.save(join(root_folder, cell_name, "EAPs", "EAP_%s.npy" % fig_name),
            LFP[:, spike_window_idx[0]:spike_window_idx[1]])

    np.save(join(root_folder, cell_name, "EAPs", "spiketime_%s.npy" % fig_name),
            cell.tvec[spike_time_idx])

    if plot or not cell_number % 10 or cell_number < 10:
        print "Vmem range: ", np.min(cell.vmem), np.max(cell.vmem)
        plot_single_cell_LFP(cell, electrode_parameters, cell_name, fig_name)


def plot_single_cell_LFP(cell, electrode_parameters,
                         cell_name, fig_name):

    ### MAKING THE ELECTRODE
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    spike_time_idx = np.argmax(cell.somav)
    spike_time = cell.tvec[spike_time_idx]
    tlim = [cell.tvec[spike_time_idx] - 15, cell.tvec[spike_time_idx] + 15]

    ### PLOTTING THE RESULTS
    elec_idx_colors = {idx: plt.cm.Reds_r(1./(len(electrode.x) + 1) * idx) for idx in range(len(electrode.x))}

    plt.close('all')
    plt.figure(figsize=(16, 9))
    # Plotting the morphology
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.suptitle('Red: Extracellular recordings')
    ax_m = plt.subplot(142, aspect=1, xlabel='x [$\mu m$]', ylabel='z [$\mu m$]', ylim=[-900, 100], xlim=[-200, 200],
                xticks=[], frameon=False, yticks=[-800, -600, -400, -200, 0])
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], c='k', clip_on=False) for idx in xrange(cell.totnsegs)]
    plt.plot(cell.xmid[0], cell.zmid[0], 'o', c="k", ms=12)

    [plt.plot(electrode.x[idx], electrode.z[idx], 'D',  c=elec_idx_colors[idx], ms=8, clip_on=False)
     for idx in xrange(len(electrode.x))]

    ax_sp = plt.subplot(141,  xlabel='Time [ms]',
                        ylabel='z [$\mu$m]',
                        title="Single-cell input spike-trains",
                       xlim=tlim, sharey=ax_m, frameon=False)

    for syn_idx, syn in enumerate(cell.synapses):

        spikes = cell.sptimeslist[syn_idx]

        c = "r" if syn.kwargs["e"] > -60. else 'b'
        ax_sp.plot(spikes, np.ones(len(spikes)) * cell.zmid[syn.idx], '.', c=c)
        ax_m.plot(cell.xmid[syn.idx], cell.zmid[syn.idx], '*', ms=7, clip_on=False, c=c)
    # Plotting the extracellular potentials
    LFP = 1000 * (electrode.LFP - electrode.LFP[:, 0, None])

    dz = electrode.z[1] - electrode.z[0]

    ax_somav = plt.subplot(143, title="somatic\nmembrane potential\nSpike time: {:0.2f} ms". format(spike_time), xlim=tlim, ylim=[-80, 50])
    ax_somav.plot(cell.tvec, cell.somav)
    ax_somav.axvline(spike_time, ls='--', c='gray')

    ax_lfp = plt.subplot(144, frameon=False, yticks=[-800, -600, -400, -200, 0],
                         sharey=ax_m,
                         title="Extracellular potential", xlim=tlim)

    normalize = np.max(np.abs((LFP[:, :] - LFP[:, 0, None])))
    for idx in range(len(electrode.z)):
        y = electrode.z[idx] + (LFP[idx] - LFP[idx, 0]) / normalize * dz
        ax_lfp.plot(cell.tvec, y, lw=1, c='k', clip_on=True)

    ax_lfp.axvline(spike_time, ls='--', c='gray')

    ax_lfp.plot([tlim[1], tlim[1]],
                [np.min(electrode.z) +dz, np.min(electrode.z)], lw=4, c='k', clip_on=False)
    ax_lfp.text(tlim[1] + 1, np.min(electrode.z) + dz/2, '{:.2f} $\mu$V'.format(normalize), clip_on=False)

    plt.savefig(join(root_folder, cell_name, '%s.png' % fig_name))

if __name__ == '__main__':

    if len(sys.argv) == 1:
        cell_name = ['hbp_L5_TTPC2_cADpyr232_1',
                     "hbp_L4_SS_cADpyr230_1",
                     "hbp_L4_SBC_bNAC219_1"][-2]
        input_type = ["waves"][0]
        initialize_population(10000, cell_name)
        single_cell_compare(cell_name=cell_name, cell_number=0, plot=True)
    else:
        single_cell_compare(
                            cell_name=sys.argv[1],
                            cell_number=int(sys.argv[2]))